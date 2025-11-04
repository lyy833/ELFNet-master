import torch 
from torch import nn
import torch.nn.functional as F
from ELFNet.dilated_conv import DilatedConvEncoder,MixedChannelConvEncoder,CI_DilatedConvEncoder
import torch.fft as fft
import math
from utils.augmentation import augment
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def load_pretrained_weights(base_model, target_model, groups):
        """
        将 ELFNet-Base 的预训练权重加载到标准 ELFNet 中,用于 one2many 跨数据集模式
        """
        print("=== 开始权重迁移: CI → VG-HCS ===")
        base_state_dict = base_model.state_dict()
        target_state_dict = target_model.state_dict()
        new_state_dict = target_state_dict.copy()
        
        # 1. 直接复制共享模块的权重 (TRD, SRD, Head等)
        print("步骤1: 迁移表示学习模块权重...")
        shared_modules = ['tfd', 'sfd', 'head']
        for key in base_state_dict:
            for module_name in shared_modules:
                if key.startswith(f'{module_name}.'):
                    if key in new_state_dict and base_state_dict[key].shape == new_state_dict[key].shape:
                        new_state_dict[key] = base_state_dict[key]
                        print(f"  ✓ 迁移 {key}")
                    elif key in new_state_dict:
                        print(f"  ⚠ 形状不匹配 {key}: {base_state_dict[key].shape} -> {new_state_dict[key].shape}")
        
        # 2. 映射特征提取器权重 (核心难点)
        print("步骤2: 映射特征提取器权重...")
        _map_feature_extractor_weights(base_state_dict, new_state_dict, groups)
        
        # 3. 加载更新后的状态字典
        target_model.load_state_dict(new_state_dict, strict=False)
        print("=== 权重迁移完成! ===")
        return target_model

    
def _map_feature_extractor_weights(base_state_dict, new_state_dict, groups):
    """
    映射特征提取器权重：从通道独立到分组混合通道
    适配你现有的 MixedChannelConvEncoder 结构
    """
    # 2.1 处理输入投影层权重
    if 'input_projection.weight' in base_state_dict and 'feature_extractor.input_fc.weight' in new_state_dict:
        base_weight = base_state_dict['input_projection.weight']  # [hidden_dim, 1]
        target_weight = new_state_dict['feature_extractor.input_fc.weight']  # [hidden_dims, input_size]
        
        # 对每个分组，使用基础权重的平均值
        group_start = 0
        for group_idx, group_vars in enumerate(groups):
            group_size = len(group_vars)
            # 使用基础权重的平均值来初始化分组权重
            avg_base_weight = base_weight.mean(dim=0, keepdim=True)  # [1, 1]
            # 扩展到分组大小
            group_weight = avg_base_weight.repeat(target_weight.size(0), group_size)
            
            # 填充到目标权重矩阵的对应位置
            target_weight[:, group_start:group_start+group_size] = group_weight
            group_start += group_size
        
        new_state_dict['feature_extractor.input_fc.weight'] = target_weight
        print(f"  ✓ 映射输入投影层权重")
    
    # 2.2 处理分组卷积权重
    for key in new_state_dict:
        if 'feature_extractor.group_convs' in key and 'weight' in key:
            # 解析分组索引
            parts = key.split('.')
            for i, part in enumerate(parts):
                if part == 'group_convs':
                    group_idx = int(parts[i+1])
                    break
            
            if group_idx < len(groups):
                group_size = len(groups[group_idx])
                
                # 寻找对应的基础编码器权重
                base_key = None
                if 'conv_layers.0' in key:
                    base_key = 'ci_encoder.conv_layers.0.weight'
                elif 'conv_layers.1' in key:
                    base_key = 'ci_encoder.conv_layers.1.weight'
                # 可以根据需要添加更多层
                
                if base_key and base_key in base_state_dict:
                    base_weight = base_state_dict[base_key]  # [out_channels, hidden_dim, kernel_size]
                    target_weight = new_state_dict[key]  # [out_channels, group_size, kernel_size]
                    
                    # 使用基础权重的平均值来初始化分组权重
                    if base_weight.dim() == 3:
                        avg_weight = base_weight.mean(dim=1, keepdim=True)  # [out_channels, 1, kernel_size]
                        expanded_weight = avg_weight.repeat(1, group_size, 1)
                        
                        if expanded_weight.shape == target_weight.shape:
                            new_state_dict[key] = expanded_weight
                            print(f"  ✓ 映射分组 {group_idx} 卷积权重: {key}")
    
    # 2.3 处理其他共享的卷积层权重
    for key in base_state_dict:
        if 'ci_encoder' in key and key.replace('ci_encoder', 'feature_extractor') in new_state_dict:
            target_key = key.replace('ci_encoder', 'feature_extractor')
            if (target_key in new_state_dict and 
                base_state_dict[key].shape == new_state_dict[target_key].shape):
                new_state_dict[target_key] = base_state_dict[key]
                print(f"  ✓ 复制共享卷积权重: {target_key}") 
                

    
class TrendFeatureDisentangler(nn.Module):
    def __init__(self, args):
        super(TrendFeatureDisentangler, self).__init__()
        self.conv_layers = nn.ModuleList()
        for k in args.kernels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(args.repr_dims, args.repr_dims // 2, kernel_size=k, padding=k-1),
                    nn.BatchNorm1d(args.repr_dims // 2),
                    nn.ReLU(inplace=True)
                )
            )
        self.residual_layer = nn.Conv1d(args.repr_dims, args.repr_dims // 2, kernel_size=1)

    def forward(self, x):
        trend_features = []
        for conv in self.conv_layers:
            trend_feature = conv(x)
            if conv[0].kernel_size[0] != 1:
                trend_feature = trend_feature[..., :-(conv[0].kernel_size[0] - 1)]
            trend_features.append(trend_feature)

        trend = torch.mean(torch.stack(trend_features, dim=0), dim=0)

        trend += self.residual_layer(x)

        return trend

### SFD的实现
class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length # 输入信号的长度
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed 频率带索引，下面设为b=1
        self.num_bands = num_bands ### 总的频带数量，下面设为1

        self.num_freqs = self.total_freqs // self.num_bands + (self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands) # 起始频率索引
        self.end = self.start + self.num_freqs  # 结束频率索引


        # case: from other frequencies
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()

    def forward(self, input):
        # input - b t d
        b, t, _ = input.shape
        # 使用快速傅里叶变换(FFT)将输入从时域转换到频域
        input_fft = fft.rfft(input, dim=1)
        # 创建一个新的元素全为0的、形状为[b,t//2+1,out_channels]的频域输出张量
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        # 对指定的频率范围应用权重和偏置
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        # 使用逆FFT将处理过的频域数据转回时域并返回。
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    # 对输入频率应用权重和偏置
    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    # 这个方法用于初始化权重和偏置参数。它使用Kaiming uniform初始化权重，并基于权重的fan-in来初始化偏置。
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)



class FeatureReducer(nn.Module):
    """
    FeatureReducer类是一个PyTorch模块，用于将输出的解耦表示转换到原始特征。
    包含多层反卷积、批归一化和ReLU激活函数。
    """
    def __init__(self, args, input_size ,  stride=1, padding=1, output_padding=0):
        super(FeatureReducer, self).__init__()
        layers = []
        in_channels = args.repr_dims

        for hidden_dim in args.reduce_hidden_dims:
            layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=args.kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_channels = hidden_dim

        layers.append(nn.ConvTranspose1d(in_channels=in_channels, out_channels=input_size, kernel_size=args.kernel_size, stride=stride, padding=padding, output_padding=output_padding))
        layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ELFNet_Base(nn.Module):
    """可迁移的预训练基模型 - 通道独立架构"""
    def __init__(self, args, device, hidden_dim=64,):
        super(ELFNet_Base, self).__init__()
        
        self.args = args
        self.device = device
        self.hidden_dim = hidden_dim
        
        # 线性投影层：将任意单变量序列投影到统一维度
        self.input_projection = nn.Linear(1, hidden_dim)
        
        # 通道独立的时序编码器 - 基于现有的DilatedConvEncoder
        self.ci_encoder = CI_DilatedConvEncoder(
            hidden_dim, args.hidden_dims, args.repr_dims, 
            args.depth, args.kernel_size
        )
        
        # 动态特征融合层 - 在forward中根据实际输入动态处理,这里只定义结构，具体维度在第一次forward时确定
        self.feature_fusion = None

        # 趋势性解耦器 - 保持原有结构
        self.tfd = TrendFeatureDisentangler(args)
        
        # 季节性解耦器 - 保持原有结构
        self.sfd = nn.ModuleList([
            BandedFourierLayer(args.repr_dims, args.repr_dims // 2, b, 1, length=args.seq_len) 
            for b in range(1)
        ])
        
        # 投影头 - 保持原有结构
        self.head = nn.Sequential(
            nn.Linear(args.repr_dims // 2, args.repr_dims // 2),
            nn.ReLU(),
            nn.Linear(args.repr_dims // 2, args.repr_dims // 2)
        )
        
        self.repr_dropout = nn.Dropout(p=0.1)
        
        # 设备配置
        if device == 'cuda:{}'.format(self.args.gpu):
            self.input_projection = self.input_projection.cuda()
            self.ci_encoder = self.ci_encoder.cuda()
            self.tfd = self.tfd.cuda()
            self.head = self.head.cuda()
            self.sfd = self.sfd.cuda()
    
    def _init_feature_fusion(self, num_vars, repr_dims):
        """根据变量数动态初始化特征融合层"""
        if num_vars > 1 and self.feature_fusion is None:
            self.feature_fusion = nn.Sequential(
                nn.Linear(repr_dims * num_vars, repr_dims),
                nn.ReLU(),
                nn.Dropout(0.1)
            ).to(self.device)
            print(f"动态初始化特征融合层: {repr_dims * num_vars} -> {repr_dims}")
        elif num_vars == 1 and self.feature_fusion is None:
            self.feature_fusion = nn.Identity()
            print("单变量情形，使用恒等映射")

    def forward(self, x):
        """
        前向传播
        x: [batch_size, seq_len, num_variables]
        """
        batch_size, seq_len, num_vars = x.shape
        
        # 动态初始化特征融合层（如果需要）
        self._init_feature_fusion(num_vars, self.args.repr_dims)

        # 通道独立处理：每个变量单独投影和编码
        var_features = []
        for var_idx in range(num_vars):
            # 提取单个变量 [batch_size, seq_len, 1]
            var_data = x[:, :, var_idx:var_idx+1]
            
            # 投影到统一维度 [batch_size, seq_len, hidden_dim]
            projected = self.input_projection(var_data)
            
            # 调整维度以适应编码器 [batch_size, hidden_dim, seq_len]
            projected = projected.transpose(1, 2)
            
            # 时序编码 [batch_size, repr_dims, seq_len]
            encoded = self.ci_encoder(projected)
            
            # 调整回 [batch_size, seq_len, repr_dims]
            encoded = encoded.transpose(1, 2)
            
            var_features.append(encoded)
        
        # 合并变量特征 [batch_size, seq_len, num_vars * repr_dims]
        combined = torch.cat(var_features, dim=-1)
        
        # 特征融合：将多变量特征融合为统一表示 [batch_size, seq_len, repr_dims]
        fused_features = self.feature_fusion(combined)
        
        # 调整维度以适应后续处理 [batch_size, repr_dims, seq_len]
        fused_features_t = fused_features.transpose(1, 2)
        
        # 趋势性成分提取
        trend_features = []
        for idx, mod in enumerate(self.tfd.conv_layers):
            out = mod(fused_features_t)
            if self.args.kernels[idx] != 1:
                out = out[..., :-(self.args.kernels[idx] - 1)]
            trend_features.append(out.transpose(1, 2))
        trend = torch.mean(torch.stack(trend_features), dim=0)
        
        # 季节性成分提取
        season = []
        for mod in self.sfd:
            out = mod(fused_features)  # b t d
            season.append(out)
        season = self.repr_dropout(season[0])
        
        return trend, season
    
    def caculate_trend_loss(self, anchor, pos, negs):
        """计算趋势性对比损失 - 保持原有实现"""
        l_pos = torch.einsum('nc,nc->n', [anchor, pos]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nkc->nk', [anchor, negs])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.args.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def convert_coeff(self, x, eps=1e-6):
        """转换复数系数 - 保持原有实现"""
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase
    
    def caculate_seasonality_loss(self, z1, z2):
        """计算季节性对比损失 - 保持原有实现"""
        B, T = z1.size(0), z1.size(1)
        z = torch.cat([z1, z2], dim=0)
        z = z.transpose(0, 1)
        sim = torch.matmul(z, z.transpose(1, 2))
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss
    
    def compute_loss(self, batch_x, batch_y, folder_path, epoch):
        """计算总损失 - 适配ELFNet-Base"""
        # 数据增强
        batch_x, positive_batch_x, negative_batch_x_list = augment(
            batch_x, batch_y, self.args.num_augment, folder_path
        )
        
        rand_idx = np.random.randint(0, batch_x.shape[1])
        
        # 计算原样本的输出
        batch_x = batch_x.to(torch.float32)
        output_t, output_s = self.forward(batch_x)
        if output_t is not None:
            output_t = F.normalize(self.head(output_t[:, rand_idx]), dim=-1)
        
        # 计算正样本的输出
        positive_batch_x = torch.from_numpy(positive_batch_x.astype('float32'))
        output_positive_t, output_positive_s = self.forward(positive_batch_x.float().to(self.device))
        if output_positive_t is not None:
            output_positive_t = F.normalize(self.head(output_positive_t[:, rand_idx]), dim=-1)
        
        # 计算负样本的输出
        output_negative_t_list = []
        for negative_batch_x in negative_batch_x_list:
            negative_batch_x = negative_batch_x.to(torch.float32)
            output_negative_t, _ = self.forward(negative_batch_x.float().to(self.device))
            if output_negative_t is not None:
                output_negative_t = F.normalize(self.head(output_negative_t[:, rand_idx]), dim=-1)
            output_negative_t_list.append(output_negative_t)
        
        output_negative_t_all = torch.stack(output_negative_t_list, dim=0).transpose(0, 1)
        
        # 季节性特征处理
        output_s = F.normalize(output_s, dim=-1)
        output_freq = fft.rfft(output_s, dim=1)
        output_positive_s = F.normalize(output_positive_s, dim=-1)
        output_positive_freq = fft.rfft(output_positive_s, dim=1)
        
        output_amp, output_phase = self.convert_coeff(output_freq)
        output_positive_amp, output_positive_phase = self.convert_coeff(output_positive_freq)
        
        # 计算损失
        trend_loss = self.caculate_trend_loss(output_t, output_positive_t, output_negative_t_all)
        seasonal_loss = self.caculate_seasonality_loss(output_amp, output_positive_amp) + \
                       self.caculate_seasonality_loss(output_phase, output_positive_phase)
        
        loss = trend_loss + self.args.alpha * seasonal_loss
        return loss


class ELFNet(nn.Module):
    def __init__(self,args, target, input_size, device, groups=None,stage2=False):
        ### 调用CLMLFNet传入的input_size就是n_c(变量数)；depth是特征提取器的深度
        super(ELFNet, self).__init__()

        self.args = args
        self.target=target # 目标变量
        self.stage2 = stage2
        self.device = device
        
        if args.compare is not None:
            ### 使用DilatedConvEncoder作为特征提取器
            self.feature_extractor = DilatedConvEncoder(
                input_size,
                args.hidden_dims,
                args.repr_dims,
                args.depth,
                args.kernel_size
            )
        else:
            self.feature_extractor = MixedChannelConvEncoder(
                args.hidden_dims,
                args.repr_dims,
                args.kernel_size,
                groups,
                args.depth
            )
        
        ### 趋势性部分的Trend Feature Disentangler使用num(kernels这个list中元素个数)个核大小为对应kernel的1D因果卷积层（没有先后顺序）构成，给定的的填充大小是kernel-1
        self.tfd = TrendFeatureDisentangler(args)


        # create the encoders
        self.head = nn.Sequential(
            nn.Linear(args.repr_dims // 2, args.repr_dims // 2),
            nn.ReLU(),
            nn.Linear(args.repr_dims // 2, args.repr_dims // 2)
        )

        ### 季节性部分使用1个BandedFourierLayer提取季节性成分
        self.sfd = nn.ModuleList(  ## nn.ModuleList详解：https://blog.csdn.net/weixin_36670529/article/details/105910767
            [BandedFourierLayer(args.repr_dims,args.repr_dims // 2, b, 1, length=args.seq_len) for b in range(1)]
        )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.feature_reducer= FeatureReducer(args,input_size)#将解耦表示的维度从repr_dims  映射回到 input_size
        
        self.projection = nn.Linear(input_size, args.c_out, bias=True)  # 新增的全连接层
        self.pool = nn.AdaptiveAvgPool1d(output_size=args.pred_len )  # 自适应平均池化层

        if device == 'cuda:{}'.format(self.args.gpu):
            self.feature_extractor = self.feature_extractor.cuda()
            self.tfd = self.tfd.cuda()
            self.head = self.head.cuda()
            self.sfd = self.sfd.cuda() 
            self.feature_reducer = self.feature_reducer.cuda()
            self.projection = self.projection.cuda()
            self.pool = self.pool.cuda()

            # 微调阶段冻结 主干特征提取器(包括attention layer、input_fc、feature_extractor )和解耦器(包括tfd和sfd)的参数
            if stage2:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                for param in self.tfd.parameters():
                    param.requires_grad = False
                for param in self.head.parameters():
                    param.requires_grad = False
                for param in self.sfd.parameters():
                    param.requires_grad = False

    def forward(self, x):# 输入的x的形状为 b,input_size,seq_len
        if  self.args.compare is not None:
            y = self.feature_extractor(x) 
        
        else:
            y = self.feature_extractor(x.transpose(1,2))
            #y = self.feature_extractor(x)
        #总的来说，下面代码片段的目标是捕获输入数据x在不同时间尺度上的趋势，并计算这些趋势的平均值。使用多尺度的方法是时间序列分析中的一种常见技巧，因为它允许模型在不同的时间尺度上捕获模式和依赖性。
        #提去趋势性成分特征
        trend = []
        for idx, mod in enumerate(self.tfd.conv_layers):
            out = mod(y)
            if self.args.kernels[idx] != 1:
                out = out[..., :-(self.args.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))
        trend = torch.mean(torch.stack(trend), dim=0)
        y = y.transpose(1, 2)

        ### 提取季节性成分的特征
        season = []
        for mod in self.sfd:
            out = mod(y)  # b t d
            season.append(out)
        season = self.repr_dropout(season[0])

        if not self.stage2:
            return trend, season
        else:
            output = torch.cat([trend, season], dim=2) # (batch_size,seq_len,repr_dims)
            output = (self.feature_reducer(output.transpose(1, 2))).transpose(1,2) # b  t input_size 

            output = self.projection(output)  # 应用全连接层，使维度从 (batch_size, t, input_size) 转换为 (batch_size, t, 1)
            output = self.pool(output.transpose(1, 2)).transpose(1, 2)  # 使用自适应平均池化调整时间步数
            return output[:, -self.args.pred_len:, :]    
    
    
    def caculate_trend_loss(self, anchor, pos, negs):
        """
        计算损失函数。这个函数用于衡量查询向量q与正向关键向量k和负向关键向量集k_negs之间的相似度。
        它通过比较查询和正向关键向量的相似度与查询和负向关键向量集的相似度之和，来识别哪些负向关键向量是最难的负样本。
        
        参数:
        anchor: 查询向量，形状为NxC。
        pos: 正向关键向量，形状为NxC。
        negs: 负向关键向量集，形状为CxL。
        
        返回:
        loss: 使用交叉熵损失函数计算的损失值。
        """
        
        # 计算查询q和正向关键向量k之间的相似度，结果为一个长度为N的一维张量
        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [anchor, pos]).unsqueeze(-1)
        
        # 计算查询q和负向关键向量集k_negs中每个关键向量的相似度，结果为一个形状为NxL的二维张量
        # negative logits: NxK
        l_neg = torch.einsum('nc,nkc->nk', [anchor, negs])
        
        # 将正向相似度和所有负向相似度连接在一起，形成一个形状为Nx(1+L)的二维张量
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # 应用温度参数T，用于调整logits的分布
        # apply temperature
        logits /= self.args.temperature
        
        # 创建一个全零标签张量，用于指示每个样本的正确类别（即正向关键向量）
        # labels: positive key indicators - first dim of each batch
        #labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        labels = torch.zeros(logits.shape[0], dtype=torch.long,device=self.device)
        
        # 使用交叉熵损失函数计算损失，其中logits是预测值，labels是标签
        loss = F.cross_entropy(logits, labels)
        
        # 返回计算得到的损失值
        return loss

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase
    
    def caculate_seasonality_loss(self, z1, z2):
        """
        计算实例对比损失。

        这个函数旨在通过比较同一实例的不同表示（z1和z2）来促进它们之间的相似性，
        同时推动不同实例之间的表示差异性。它通过构建一个相似性矩阵，并从中提取出
        对比损失来实现这一目标。

        参数:
        z1: Tensor, 形状为(B, T, C)的实例表示1，其中B是批次大小，T是序列长度，C是特征维度。
        z2: Tensor, 形状为(B, T, C)的实例表示2，应与z1对应。

        返回值:
        loss: Tensor, 形状为()的标量Tensor，表示实例对比损失。
        """
        # 合并z1和z2，以便在批次维度上进行对比
        B, T = z1.size(0), z1.size(1)
        # 使用torch.cat将z1和z2沿批次维度B合并，得到形状为2B x T x C的张量z。
        z = torch.cat([z1, z2], dim=0)  
        # 转置z，以便在序列长度维度上进行矩阵乘法
        # 将z进行转置，使其形状变为T x 2B x C
        z = z.transpose(0, 1)  
        # 计算z与自身转置的矩阵乘积，得到相似性矩阵
        # 使用torch.matmul函数计算z与其转置(T x C x 2B)的点积，得到形状为T x 2B x 2B的相似性矩阵sim。这个矩阵表示每一对实例之间的相似性。
        sim = torch.matmul(z, z.transpose(1, 2)) 
        # 通过取下三角和上三角，构造对比学习所需的logits矩阵
        # 下面两句使用torch.tril和torch.triu函数来取sim矩阵的下三角部分和上三角部分，
        # 并进行拼接，得到形状为T x 2B x (2B-1)的对数矩阵logits。
        # 这样，每一行的对数矩阵都不会包含对角线上的元素（自身的相似性）。 
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        # 对logits应用负对数 softmax，以得到对比损失的合适形式
        # 使用-F.log_softmax对logits进行归一化处理，使每行的所有元素之和为1。
        logits = -F.log_softmax(logits, dim=-1)
        
        # 计算每个实例的平均对比损失
        # 对于每一个实例，在logits中找到与其对应的正例（相似的实例）和负例（不相似的实例）的相似性得分。
        # 使用这些得分来计算总体的对比损失。
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss
    


    def compute_loss(self,batch_x,batch_y,folder_path,epoch):
        batch_x, positive_batch_x, negative_batch_x_list= augment(batch_x,batch_y,self.args.num_augment,folder_path)
        
        rand_idx = np.random.randint(0, batch_x.shape[1]) 
        
        # 计算原样本的趋势性输出
        batch_x = batch_x.to(torch.float32)
        batch_x = batch_x.transpose(1, 2).to(self.device)
        output_t, output_s= self.forward(batch_x) # (b,seq_len,repr_dims/2) 
        if output_t is not None:
            output_t = F.normalize(self.head(output_t[:, rand_idx]), dim=-1)
        
        # 计算正样本的趋势性输出
        positive_batch_x = torch.from_numpy(positive_batch_x.astype('float32'))
        positive_batch_x = positive_batch_x.transpose(1, 2)
        output_positive_t, output_positive_s= self.forward(positive_batch_x.float().to(self.device)) # (b,seq_len,repr_dims/2) 
        if output_positive_t is not None:
            output_positive_t = F.normalize(self.head(output_positive_t[:, rand_idx]), dim=-1) #(batch_size,seq_len,repr_dims/2)
        
        # 计算所有负样本的趋势性输出
        output_negative_t_list = []
        for negative_batch_x in negative_batch_x_list:
            # 计算每个负样本的趋势性输出
            negative_batch_x = negative_batch_x.to(torch.float32)
            negative_batch_x = negative_batch_x.transpose(1, 2)
            output_negative_t, _= self.forward(negative_batch_x.float().to(self.device)) # (b,seq_len,repr_dims/2) 
            if output_negative_t is not None:
                output_negative_t = F.normalize(self.head(output_negative_t[:, rand_idx]), dim=-1) #(batch_size,repr_dims/2)
            output_negative_t_list.append(output_negative_t)
        output_negative_t_all = torch.stack(output_negative_t_list,dim=0) # (k,b,repr_dims/2)   其中k是负样本个数，k = num_augment
        output_negative_t_all = output_negative_t_all.transpose(0,1) # (b,k,repr_dims/2)
        #使用离散傅里叶变换等处理得到的原样本和正样本季节性输出矩阵
        output_s = F.normalize(output_s, dim=-1) # (b,seq_len,repr_dims/2)
        output_freq = fft.rfft(output_s, dim=1) # (b,seq_len//2+1,repr_dims/2)
        output_positive_s = F.normalize(output_positive_s, dim=-1) 
        output_positive_freq = fft.rfft(output_positive_s, dim=1)       
        # 将原/正样本的季节性输出转换为样本的幅度和相位
        output_amp, output_phase= self.convert_coeff(output_freq)
        output_positive_amp, output_positive_phase= self.convert_coeff(output_positive_freq)
        
        # 计算季节性对比损失和趋势性对比损失
        trend_loss = self.caculate_trend_loss(output_t,output_positive_t,output_negative_t_all)
        seasonal_loss = self.caculate_seasonality_loss(output_amp, output_positive_amp)+ self.caculate_seasonality_loss(output_phase,output_positive_phase)
        
        # 计算总对比损失
        loss =   trend_loss + self.args.alpha * seasonal_loss
        return loss
    

    