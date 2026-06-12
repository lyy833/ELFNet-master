import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from torch.nn.init import xavier_normal_, constant_

# 基础模块定义（从原始代码迁移）
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, T, D]
        return self.pe[:, :x.size(1), :]

class TransformerBlock(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, attn_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.enable_res_parameter = enable_res_parameter
        if enable_res_parameter:
            self.res_param1 = nn.Parameter(torch.ones(1))
            self.res_param2 = nn.Parameter(torch.ones(1))

    def forward(self, x, mask=None):
        # x: [B, T, D]
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        if self.enable_res_parameter:
            x = self.norm1(x + self.res_param1 * self.dropout(attn_out))
        else:
            x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        if self.enable_res_parameter:
            x = self.norm2(x + self.res_param2 * self.dropout(ffn_out))
        else:
            x = self.norm2(x + self.dropout(ffn_out))
        return x

class CrossAttnTRMBlock(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, attn_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model)
        )
        self.enable_res_parameter = enable_res_parameter
        if enable_res_parameter:
            self.res_param1 = nn.Parameter(torch.ones(1))
            self.res_param2 = nn.Parameter(torch.ones(1))

    def forward(self, visible, mask_token):
        # visible: [B, T_v, D], mask_token: [B, T_m, D]
        attn_out, _ = self.cross_attn(mask_token, visible, visible)
        if self.enable_res_parameter:
            mask_token = self.norm1(mask_token + self.res_param1 * attn_out)
        else:
            mask_token = self.norm1(mask_token + attn_out)
        ffn_out = self.ffn(mask_token)
        if self.enable_res_parameter:
            mask_token = self.norm2(mask_token + self.res_param2 * ffn_out)
        else:
            mask_token = self.norm2(mask_token + ffn_out)
        return mask_token

class Encoder(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, layers, enable_res_parameter, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout)
            for _ in range(layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Tokenizer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.center = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [B, T, D]
        probs = self.center(x)
        # Gumbel softmax
        ret = F.gumbel_softmax(probs, hard=False)
        return ret  # 返回概率，索引在 compute_loss 中处理

class Regressor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter)
            for _ in range(layers)
        ])

    def forward(self, rep_visible, rep_mask_token):
        for layer in self.layers:
            rep_mask_token = layer(rep_visible, rep_mask_token)
        return rep_mask_token


class TimeMAE(nn.Module):
    """
    Self-supervised TimeMAE model for forecasting.
    stage=1: pretrain, compute_loss returns reconstruction loss.
    stage=2: finetune, forward returns prediction [B, pred_len, 1].
    """
    def __init__(self, configs, stage=1):
        super().__init__()
        self.configs = configs
        self.stage = stage
        self.d_model = configs.repr_dims          # 使用 repr_dims 作为 d_model
        self.attn_heads = configs.n_heads
        self.layers = configs.e_layers
        self.reg_layers = getattr(configs, 'reg_layers', 2)
        self.dropout = configs.dropout
        self.enable_res_parameter = getattr(configs, 'enable_res_parameter', True)
        self.vocab_size = getattr(configs, 'vocab_size', 100)  # 需要配置
        self.mask_ratio = getattr(configs, 'mask_ratio', 0.4)
        self.wave_length = configs.wave_length
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.pred_len = configs.pred_len
        self.target_idx = configs.finetune_target_idx
        self.momentum = getattr(configs, 'momentum', 0.999)

        # 计算 patch 数量
        self.num_patch = self.seq_len // self.wave_length
        assert self.seq_len % self.wave_length == 0, "seq_len must be divisible by wave_length"
        self.mask_len = int(self.mask_ratio * self.num_patch)

        # 位置编码
        self.position = PositionalEmbedding(self.num_patch, self.d_model)

        # 输入投影
        self.input_projection = nn.Conv1d(
            self.enc_in, self.d_model,
            kernel_size=self.wave_length,
            stride=self.wave_length
        )

        # 编码器
        d_ffn = 4 * self.d_model
        self.encoder = Encoder(self.d_model, self.attn_heads, d_ffn, self.layers, self.enable_res_parameter, self.dropout)
        self.momentum_encoder = Encoder(self.d_model, self.attn_heads, d_ffn, self.layers, self.enable_res_parameter, self.dropout)
        self.copy_weight()  # 初始化动量编码器

        # Tokenizer
        self.tokenizer = Tokenizer(self.d_model, self.vocab_size)

        # Regressor
        self.reg = Regressor(self.d_model, self.attn_heads, d_ffn, self.enable_res_parameter, self.reg_layers)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(self.d_model))

        
        self.apply(self._init_weights)

        # 微调阶段：创建预测头，并使用小标准差初始化
        self.prediction_head = nn.Linear(self.d_model, self.pred_len)
        nn.init.normal_(self.prediction_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.prediction_head.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data.copy_(param_a.data)

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def _pretrain_forward(self, x):
        """
        x: [B, T, D] 原始序列
        返回:
            rep_mask: [B, mask_len, D] 真实被mask部分的表示（来自动量编码器）
            rep_pred: [B, mask_len, D] 预测的表示
            token_pred: [B, mask_len, vocab_size] 预测的token概率
            token_true: [B, mask_len] 真实的token索引
        """
        # 输入投影: [B, T, D] -> [B, num_patch, d_model]
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()

        # 获取 token（离散化）
        token_probs = self.tokenizer(x)  # [B, num_patch, vocab_size]
        token_true = token_probs.argmax(dim=-1)  # [B, num_patch]

        # 添加位置编码
        x = x + self.position(x)

        # 生成 mask 索引
        index = list(range(self.num_patch))
        random.shuffle(index)
        v_index = index[:-self.mask_len]
        m_index = index[-self.mask_len:]

        visible = x[:, v_index, :]            # [B, vis_len, D]
        mask_original = x[:, m_index, :]      # [B, mask_len, D]
        tokens_masked = token_true[:, m_index] # [B, mask_len]

        # 准备 mask token 表示
        rep_mask_token = self.mask_token.unsqueeze(0).unsqueeze(0).repeat(x.size(0), self.mask_len, 1)
        rep_mask_token = rep_mask_token + self.position(x)[:, m_index, :]

        # 编码可见部分
        rep_visible = self.encoder(visible)   # [B, vis_len, D]

        # 动量编码器提取被 mask 部分的真实表示（无梯度）
        with torch.no_grad():
            rep_mask_true = self.momentum_encoder(mask_original)  # [B, mask_len, D]

        # 通过 regressor 预测被 mask 部分的表示
        rep_mask_pred = self.reg(rep_visible, rep_mask_token)     # [B, mask_len, D]

        # 预测 token
        token_pred = self.tokenizer.center(rep_mask_pred)        # [B, mask_len, vocab_size]

        return rep_mask_true, rep_mask_pred, token_pred, tokens_masked

    def compute_loss(self, x):
        """
        预训练损失：表示重建 MSE + token 交叉熵
        """
        rep_mask_true, rep_mask_pred, token_pred, token_true = self._pretrain_forward(x)

        # 表示重建损失（MSE）
        repr_loss = F.mse_loss(rep_mask_pred, rep_mask_true)

        # token 预测损失（交叉熵）
        token_loss = F.cross_entropy(
            token_pred.reshape(-1, self.vocab_size),
            token_true.reshape(-1)
        )

        # 总损失（可调整权重，原论文中可能没有明确，这里简单相加）
        loss = repr_loss + token_loss

        # 动量更新
        self.momentum_update()

        return loss

    def forward(self, x,x_mark=None, dec_inp=None, y_mark=None):
        """
        微调阶段前向传播
        x: [B, T, D]
        返回: [B, pred_len, 1]
        """
        if x.shape[1] != self.seq_len:
            x = x.transpose(1,2)
        # 输入投影
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()  # [B, num_patch, d_model]
        x = x + self.position(x)

        # 编码器
        rep = self.encoder(x)  # [B, num_patch, d_model]

        # 池化（取均值）得到序列表示
        seq_rep = rep.mean(dim=1)  # [B, d_model]

        # 预测头
        out = self.prediction_head(seq_rep)  # [B, pred_len]

        # 添加目标变量维度（假设单变量预测）
        out = out.unsqueeze(-1)  # [B, pred_len, 1]

        return out

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn, freeze_encoder=True):
        """加载预训练权重，并冻结所有预训练模块（不包含 prediction_head）"""
        state_dict = torch.load(fn, map_location=next(self.parameters()).device)
        # 去除可能的 DataParallel 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # strict=False 允许缺少新模块（如 prediction_head）的键
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print("Missing keys (expected for new layers):", missing)
        print("Unexpected keys:", unexpected)

        # 冻结所有非 prediction_head 的参数
        for name, param in self.named_parameters():
            if not name.startswith('prediction_head'):
                param.requires_grad = False