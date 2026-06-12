import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from typing import List
from einops import reduce, rearrange
from torch.fft import rfft
import torch.fft as fft


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, extract_layers=None):
        super().__init__()

        if extract_layers is not None:
            assert len(channels) - 1 in extract_layers

        self.extract_layers = extract_layers
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        if self.extract_layers is not None:
            outputs = []
            for idx, mod in enumerate(self.net):
                x = mod(x)
                if idx in self.extract_layers:
                    outputs.append(x)
            return outputs
        return self.net(x)



def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)



class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs


        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)


class CoSTEncoder(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial'):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
        )

        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
        )

    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims
        if x.shape[2] != self.input_dims:
            x = x.transpose(1,2)
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
        x = self.feature_extractor(x)  # B x Co x T

        if tcn_output:
            return x.transpose(1, 2)

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)  # b d t
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))  # b t d
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )

        x = x.transpose(1, 2)  # B x T x Co

        season = []
        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]

        return trend, self.repr_dropout(season)



class CoST(nn.Module):
    """
    Self-supervised CoST model for forecasting.
    stage=1: pretrain, compute_loss returns contrastive loss.
    stage=2: finetune, forward returns prediction [B, pred_len, 1].
    """
    def __init__(self, configs, stage=1):
        super().__init__()
        self.configs = configs
        self.stage = stage
        self.input_dims = configs.enc_in
        self.output_dims = configs.repr_dims          # 趋势/季节总的维度
        self.hidden_dims = configs.hidden_dims
        self.depth = configs.depth
        self.kernels = configs.kernels                # 需要包含在configs中，例如 [1,2,4,8]
        self.alpha = getattr(configs, 'alpha', 0.05)  # 季节损失权重
        self.K = getattr(configs, 'K', 256)           # 队列大小
        self.m = getattr(configs, 'm', 0.999)         # 动量更新系数
        self.T = getattr(configs, 'T', 0.07)          # 温度参数
        self.aug_p = getattr(configs, 'aug_p', 0.5)
        self.target_idx = configs.finetune_target_idx
        self.pred_len = configs.pred_len
        self.max_train_length = configs.seq_len
        self.sigma = getattr(configs, 'aug_sigma', 0.5)  # 增强噪声强度

        # 初始化编码器（查询和键）
        self.encoder_q = CoSTEncoder(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            kernels=self.kernels,
            length=self.max_train_length,
            hidden_dims=self.hidden_dims,
            depth=self.depth
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)

        # 投影头（用于趋势对比）
        self.head_q = nn.Sequential(
            nn.Linear(self.output_dims//2, self.output_dims//2),
            nn.ReLU(),
            nn.Linear(self.output_dims//2, self.output_dims//2)
        )
        self.head_k = copy.deepcopy(self.head_q)

        # 初始化动量编码器参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 队列（用于趋势对比）
        self.register_buffer('queue', F.normalize(torch.randn(self.output_dims//2, self.K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # 微调阶段的预测头 
        self.projection = nn.Linear(self.output_dims, self.pred_len)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.001)
        if self.projection.bias is not None:
            nn.init.constant_(self.projection.bias, 0.0)

    def _augment_batch(self, x):
        """
        对输入批次应用增强（scale, shift, jitter），保持序列长度不变。
        增强策略与官方 PretrainDataset 完全一致。
        x: [B, T, D] 其中 T 应等于 self.max_train_length
        """
        B, T, D = x.shape
        # 克隆两份作为两个视图的初始值
        x_q = x.clone()
        x_k = x.clone()

        # 定义单个视图的增强函数（与官方 transform 顺序一致）
        def _transform(y):
            # scale: 每个通道乘以随机因子 (randn * sigma + 1)
            if torch.rand(1).item() < self.aug_p:
                scale_factor = torch.randn(D, device=y.device) * self.sigma + 1
                y = y * scale_factor.unsqueeze(0).unsqueeze(0)   # [1,1,D] 广播
            # shift: 每个通道加上随机偏置 (randn * sigma)
            if torch.rand(1).item() < self.aug_p:
                shift_bias = torch.randn(D, device=y.device) * self.sigma
                y = y + shift_bias.unsqueeze(0).unsqueeze(0)
            # jitter: 整个张量加高斯噪声
            if torch.rand(1).item() < self.aug_p:
                noise = torch.randn_like(y) * self.sigma
                y = y + noise
            return y

        x_q = _transform(x_q)
        x_k = _transform(x_k)
        return x_q, x_k

    def _compute_contrastive_loss(self, q, k, k_negs):
        """InfoNCE loss for trend representations"""
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)          # [B,1]
        l_neg = torch.einsum('nc,ck->nk', [q, k_negs])                  # [B, K]
        logits = torch.cat([l_pos, l_neg], dim=1)                       # [B, 1+K]
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)

    def _convert_coeff(self, x, eps=1e-6):
        """Convert complex tensor to amplitude and phase"""
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase

    def _instance_contrastive_loss(self, z1, z2):
        """Instance-level contrastive loss for seasonal representations"""
        B, T, C = z1.shape
        z = torch.cat([z1, z2], dim=0)                 # [2B, T, C]
        z = z.transpose(0, 1)                           # [T, 2B, C]
        sim = torch.matmul(z, z.transpose(1, 2))        # [T, 2B, 2B]
        # 生成掩码，去除对角线
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + \
                 torch.triu(sim, diagonal=1)[:, :, 1:]  # [T, 2B, 2B-1]
        logits = -F.log_softmax(logits, dim=-1)
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新键编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def compute_loss(self, x):
        """
        预训练阶段对比损失
        x: [B, T, D] 原始批次数据（张量）
        """
        # 生成两个增强视图
        x_q, x_k = self._augment_batch(x)

        # 随机选择一个时间步用于趋势对比
        rand_idx = np.random.randint(0, x_q.shape[1])

        # 查询视图
        q_t, q_s = self.encoder_q(x_q)                # q_t, q_s: [B, T, D] 或 [B, T, D]? 假设均为 [B, T, D]
        if q_t is not None:
            q_t_feat = F.normalize(self.head_q(q_t[:, rand_idx]), dim=-1)  # [B, D]

        # 动量更新键编码器
        self._momentum_update_key_encoder()
        with torch.no_grad():
            k_t, k_s = self.encoder_k(x_k)
            if k_t is not None:
                k_t_feat = F.normalize(self.head_k(k_t[:, rand_idx]), dim=-1)  # [B, D]

        # 趋势对比损失
        loss = self._compute_contrastive_loss(q_t_feat, k_t_feat, self.queue.clone().detach())
        self._dequeue_and_enqueue(k_t_feat)

        # 季节对比损失
        q_s_norm = F.normalize(q_s, dim=-1)            # [B, T, D]
        # 使用同一批次的另一个增强视图（用encoder_q提取，无梯度）
        with torch.no_grad():
            _, k_s_other = self.encoder_q(x_k)         # 使用查询编码器，避免重复更新
            k_s_other = F.normalize(k_s_other, dim=-1)

        # 频域变换
        q_s_freq = rfft(q_s_norm, dim=1)
        k_s_freq = rfft(k_s_other, dim=1)
        q_amp, q_phase = self._convert_coeff(q_s_freq)
        k_amp, k_phase = self._convert_coeff(k_s_freq)

        seasonal_loss = (self._instance_contrastive_loss(q_amp, k_amp) +
                         self._instance_contrastive_loss(q_phase, k_phase)) / 2
        loss += self.alpha * seasonal_loss

        return loss

    # ------------------ 微调前向 ------------------
    def forward(self, x,x_mark=None, dec_inp=None, y_mark=None):
        """
        微调阶段前向传播
        x: [B, T, D] 原始时间序列
        返回: [B, pred_len, 1]
        """
        trend, seasonal = self.encoder_q(x)            # [B, T, D]
        # 取最后一个时间步
        trend_last = trend[:, -1, :]                   # [B, D]
        seasonal_last = seasonal[:, -1, :]             # [B, D]
        feat = torch.cat([trend_last, seasonal_last], dim=-1)  # [B, 2D]
        out = self.projection(feat)                    # [B, pred_len]
        out = out.unsqueeze(-1)                         # [B, pred_len, 1]
        return out

    # ------------------ 保存与加载 ------------------
    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        """加载预训练权重到 encoder_q 和 head_q（子模块级别）"""
        full_state_dict = torch.load(fn, map_location=next(self.parameters()).device)

        # 1. 加载 encoder_q
        encoder_q_dict = {}
        for k, v in full_state_dict.items():
            if k.startswith('encoder_q.'):
                new_key = k[len('encoder_q.'):]  # 去除前缀
                encoder_q_dict[new_key] = v

        # 2. 加载 head_q
        head_q_dict = {}
        for k, v in full_state_dict.items():
            if k.startswith('head_q.'):
                new_key = k[len('head_q.'):]  # 去除前缀
                head_q_dict[new_key] = v

        for param in self.encoder_q.parameters():
            param.requires_grad = False
        for param in self.head_q.parameters():
            param.requires_grad = False