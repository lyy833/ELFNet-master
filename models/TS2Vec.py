import torch.nn as nn
import torch
import torch.nn.functional as F
from layers.dilated import DilatedConvEncoder
from utils.masking import generate_binomial_mask,generate_continuous_mask
import numpy as np


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        if x.shape[2] != self.input_dims:
            x=x.transpose(1,2)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
        

# ------------------------------------------------------------
# 辅助函数（官方 TS2Vec 中的工具）
# ------------------------------------------------------------
def take_per_row(A, indx, num_elem):
    """
    从 A 的每一行中取从 indx[i] 开始的连续 num_elem 个元素
    A: [B, T, D]
    indx: [B]
    num_elem: int
    return: [B, num_elem, D]
    """
    all_indx = indx[:, None] + torch.arange(num_elem, device=A.device)[None, :]
    return torch.gather(A, 1, all_indx.unsqueeze(-1).expand(-1, -1, A.size(-1)))


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    """
    分层对比损失（官方 TS2Vec 实现）
    z1, z2: [B, T, D]
    """
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 2 ** d < z1.size(1):
                loss += alpha * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss

def instance_contrastive_loss(z1, z2):
    """
    实例级对比损失（InfoNCE）
    z1, z2: [B, T, D] 或 [B, D]
    """
    B, T, D = z1.shape
    z1 = z1.reshape(B * T, D) if T > 1 else z1.squeeze(1)
    z2 = z2.reshape(B * T, D) if T > 1 else z2.squeeze(1)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = torch.matmul(z1, z2.T)  # [B*T, B*T]
    sim = sim - torch.max(sim, dim=1, keepdim=True)[0]
    logits = sim / 0.1  # 温度固定为 0.1
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def temporal_contrastive_loss(z1, z2):
    """
    时间级对比损失（同一实例的不同时间步）
    """
    B, T, D = z1.shape
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = torch.bmm(z1, z2.transpose(1, 2))  # [B, T, T]
    sim = sim.view(B * T, T)  # [B*T, T]
    sim = sim - torch.max(sim, dim=1, keepdim=True)[0]
    logits = sim / 0.1
    labels = torch.arange(T, device=z1.device).repeat(B)
    loss = F.cross_entropy(logits, labels)
    return loss



# ------------------------------------------------------------
# 主类 TS2Vec
# ------------------------------------------------------------
class TS2Vec(nn.Module):
    """
    统一 TS2Vec 模型，支持预训练和微调。
    stage=1: 预训练模式，forward 返回编码表示，compute_loss 计算自监督损失
    stage=2: 微调模式，forward 返回预测值 [B, pred_len, 1]
    """
    def __init__(self, configs, stage=1,freeze_encoder=True):
        super().__init__()
        self.configs = configs
        self.freeze_encoder = freeze_encoder
        self.stage = stage
        self.input_dims = configs.enc_in
        self.output_dims = configs.repr_dims
        self.hidden_dims = configs.hidden_dims
        self.depth = configs.depth
        self.target_idx = configs.finetune_target_idx
        self.pred_len = configs.pred_len
        self.max_train_length = configs.seq_len
        self.temporal_unit = getattr(configs, 'temporal_unit', 0)

        # 编码器
        self._net = TSEncoder(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth
        )
        # 微调模式：添加预测头（使用最后一个时间步的表示）
        self.projection = nn.Linear(self.output_dims, self.pred_len)
        
        

    def forward(self, x,x_mark=None, dec_inp=None, y_mark=None):
        """
        x: [B, T, D]
        返回:
            stage=1: 编码表示 [B, T, output_dims]
            stage=2: 预测值 [B, pred_len, 1]
        """
        reprs = self._net(x)  # [B, T, output_dims]
        if self.stage == 1:
            return reprs
        else:
            last_repr = reprs[:, -1, :]  # [B, output_dims]
            out = self.projection(last_repr)  # [B, pred_len]
            out = out.unsqueeze(-1)  # [B, pred_len, 1]
            return out

    def compute_loss(self, batch_x):
        """
        预训练阶段的自监督对比损失计算。
        batch_x: [B, T, D] 原始批次数据（张量）
        返回: 损失标量
        """
        # 设备处理
        x = batch_x.float().to(next(self._net.parameters()).device)

        # 官方 TS2Vec 增强：随机裁剪两个重叠视图
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(
            low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0)
        )
        crop_offset = torch.tensor(crop_offset, device=x.device)

        # 视图1
        x1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
        x1 = x1[:, -crop_l:]  # 取最后 crop_l 长度

        # 视图2
        x2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
        x2 = x2[:, :crop_l]   # 取前 crop_l 长度

        # 获取表示
        z1 = self._net(x1)  # [B, crop_l, output_dims]
        z2 = self._net(x2)  # [B, crop_l, output_dims]

        # 分层对比损失
        loss = hierarchical_contrastive_loss(
            z1, z2,
            alpha=0.5,
            temporal_unit=self.temporal_unit
        )
        return loss

    def save(self, fn):
        """保存编码器权重（预训练后调用）"""
        torch.save(self._net.state_dict(), fn)

    def load(self, fn):
        """加载编码器权重"""
        full_state_dict = torch.load(fn, map_location=next(self._net.parameters()).device)
        # 提取 _net 对应的参数，去除 '_net.' 前缀
        net_state_dict = {}
        for k, v in full_state_dict.items():
            if k.startswith('_net.'):
                net_state_dict[k[5:]] = v
        self._net.load_state_dict(net_state_dict)
        if self.freeze_encoder:
            for param in self._net.parameters():
                param.requires_grad = False