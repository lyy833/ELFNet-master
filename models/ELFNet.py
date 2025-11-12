import torch 
from torch import nn
import torch.nn.functional as F
from layers.dilated_conv import ConvBlock
import torch.fft as fft
import math
from utils.augmentation import augment
import warnings

warnings.filterwarnings('ignore')
class MultiScaleContextExtractor(nn.Module):
    """
    多尺度上下文提取器,使用卷积处理多通道,用于双向调制机制
    """
    def __init__(self, feature_dim, num_scales=3):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 多尺度池化
        self.pool_layers = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),  # 全局 -> [B, C, 1]
            nn.AdaptiveAvgPool1d(2),  # 中等 -> [B, C, 2] 
            nn.AdaptiveAvgPool1d(4)   # 细粒度 -> [B, C, 4]
        ])
        
        # 使用1D卷积处理每个尺度的特征，保持通道维度
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),  # 全局: [B, C, 1] -> [B, C, 1]
            nn.Conv1d(feature_dim, feature_dim, kernel_size=2),  # 中等: [B, C, 2] -> [B, C, 1]
            nn.Conv1d(feature_dim, feature_dim, kernel_size=4)   # 细粒度: [B, C, 4] -> [B, C, 1]
        ])
        
        # 尺度注意力权重
        self.scale_attention = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_scales),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        x: [B, T, C] 输入特征
        返回: [B, C] 多尺度融合的上下文向量
        """
        B, T, C = x.shape
        x_transposed = x.transpose(1, 2)  # [B, C, T]
        
        scale_features = []
        for pool, conv in zip(self.pool_layers, self.scale_convs):
            if T >= pool.output_size:
                # 池化
                pooled = pool(x_transposed)  # [B, C, L]
                
                # 使用卷积处理（不需要转置）
                # 对于 kernel_size > pooled_length 的情况，使用适当的填充
                L = pooled.size(2)
                if conv.kernel_size[0] > L:
                    # 如果卷积核大于序列长度，使用较小的核
                    temp_conv = nn.Conv1d(C, C, kernel_size=L, padding=0).to(x.device)
                    projected = temp_conv(pooled)  # [B, C, 1]
                else:
                    projected = conv(pooled)  # [B, C, 1]
                
                scale_features.append(projected.squeeze(-1))  # [B, C]
            else:
                # 如果序列长度不够，使用零填充
                scale_features.append(torch.zeros(B, C, device=x.device))
        
        # 拼接多尺度特征 [B, C×3]
        concatenated = torch.cat(scale_features, dim=-1)  # [B, 3×C]
        
        # 计算尺度权重
        scale_weights = self.scale_attention(concatenated)  # [B, 3]
        
        # 加权融合
        weighted_sum = torch.zeros(B, C, device=x.device)
        for i, feat in enumerate(scale_features):
            weight = scale_weights[:, i].unsqueeze(-1)  # [B, 1]
            weighted_sum += weight * feat
        
        return weighted_sum  # [B, C]

class CoupledGatingUnit(nn.Module):
    """
    基于耦合门控单元的季节-趋势性表示双向调制器。
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 多尺度上下文提取
        self.context_extractor = MultiScaleContextExtractor(feature_dim)
        
        # 趋势→季节的缩放门控
        self.trend_to_season_scale = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()
        )
        
        # 季节→趋势的偏移门控  
        self.season_to_trend_shift = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()
        )
        
        # 可学习的调制强度
        self.scale_factor = nn.Parameter(torch.tensor(0.1))
        self.shift_factor = nn.Parameter(torch.tensor(0.1))

    def forward(self, trend, season):
        """
        改进的双向调制 - 考虑LayerNorm时机
        """
        B, T, C = trend.shape
        
        # 使用多尺度上下文提取
        trend_context = self.context_extractor(trend)  # [B, C]
        season_context = self.context_extractor(season)  # [B, C]
        
        # 趋势→季节：缩放调制
        g_trend_to_season = self.trend_to_season_scale(trend_context)  # [B, C]
        g_trend_to_season = g_trend_to_season.unsqueeze(1) * self.scale_factor  # [B, 1, C]
        
        # 季节→趋势：偏移调制  
        g_season_to_trend = self.season_to_trend_shift(season_context)  # [B, C]
        g_season_to_trend = g_season_to_trend.unsqueeze(1) * self.shift_factor  # [B, 1, C]
        
        # 应用调制
        modulated_season = season + g_trend_to_season * season
        modulated_trend = trend + g_season_to_trend
        
        # 关键：调制后立即应用LayerNorm保持量级稳定
        modulated_season = F.layer_norm(modulated_season, (C,))
        modulated_trend = F.layer_norm(modulated_trend, (C,))
        
        return modulated_trend, modulated_season



class MixedChannelConvEncoder(nn.Module):
    """
    Feature Extractor 实现。
    """
    def __init__(self,hidden_dims, repr_dims, kernel_size, groups, depth,device):
        super().__init__()

        self.device = device
        self.groups = groups
        self.hidden_dims = hidden_dims
        # Define conv layers for all groups using dilated convolution
        self.group_convs = nn.ModuleList([
            nn.Sequential(*[
                ConvBlock(
                    hidden_dims if i > 0 else len(group)*hidden_dims,
                    hidden_dims,
                    repr_dims,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    first=(i==0),
                    final=False,
                    mixed=True
                )
                for i in range(depth-1) ### 从0到depth-1的层，依次进行卷积操作，并使用gelu激活函数
              ])
            for group in groups
        ])

        # Final convolution to merge features
        total_channels = hidden_dims * len(groups)
        self.final_conv = nn.Conv1d(total_channels, repr_dims, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, total_features  = x.size()
        
        conv_results = []

        # Process each group using dilated convolution
        for idx, group in enumerate(self.groups):
            # 计算组内所有变量对应的特征块
            group_size = len(group)
            group_features = torch.zeros(batch_size, seq_len, group_size * self.hidden_dims)
            
            for pos, var_idx in enumerate(group):
                start_src = var_idx * self.hidden_dims
                end_src = (var_idx + 1) * self.hidden_dims
                start_tgt = pos * self.hidden_dims
                end_tgt = (pos + 1) * self.hidden_dims
                
                group_features[:, :, start_tgt:end_tgt] = x[:, :, start_src:end_src]
            
            group_features = group_features.transpose(1, 2).to(self.device)
            conv_result = self.group_convs[idx](group_features)
            conv_results.append(conv_result)
                
        # Concatenate all convolution results along the channel dimension
        conv_results = torch.cat(conv_results, dim=1)  # Shape: (batch_size, total_channels, seq_len)
        
        # Apply final convolution
        output = self.final_conv(conv_results)
        
        return output # Shape: (batch_size, channels, seq_len)


class TrendRepresentationDisentangler(nn.Module):
    """
    TRD的实现。
    """
    def __init__(self, args):
        super(TrendRepresentationDisentangler, self).__init__()
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



class BandedFourierLayer(nn.Module):
    """
    SRD实现的核心组件。
    """
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

class MultiBandSeasonalDisentangler(nn.Module):
    """
    SRD的实现。
    """
    def __init__(self, in_channels, out_channels, num_bands=3, length=201):
        super().__init__()
        self.length = length
        self.num_bands = num_bands
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 每个频带独立的傅里叶层
        self.band_layers = nn.ModuleList([
            BandedFourierLayer(in_channels, out_channels, b, num_bands, length=length)
            for b in range(num_bands)
        ])
        
        # 轻量级频带注意力（无外部上下文）
        self.band_attention = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, num_bands),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        x: [B, T, D] 输入特征
        返回: [B, T, out_channels] 季节性表示，与原来维度一致
        """
        B, T, D = x.shape
        
        # 多频带分解
        band_outputs = []
        for layer in self.band_layers:
            band_out = layer(x)  # [B, T, out_channels]
            band_outputs.append(band_out)
        
        # 自适应频带融合
        # 使用输入特征的统计信息作为注意力上下文
        context = x.mean(dim=1)  # [B, D] - 时序平均作为上下文
        attn_weights = self.band_attention(context)  # [B, num_bands]
        
        # 加权融合
        seasonal_output = torch.zeros_like(band_outputs[0])
        for i, band_out in enumerate(band_outputs):
            weight = attn_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            seasonal_output += weight * band_out
        
        return seasonal_output



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


class ELFNet(nn.Module):
    def __init__(self,args,  device, stage2=False):
        super(ELFNet, self).__init__()

        self.args = args
        self.stage2 = stage2
        self.device = device

        # 动态初始化的输入投影层列表
        self.input_projection_list = None

        # 特征提取器
        self.feature_extractor = None # 变量分组可能发生变化，前向传播中在具体确定
        
        # Trend Representation Disentangler使用num(kernels这个list中元素个数)个核大小为对应kernel的1D因果卷积层（没有先后顺序）构成，给定的的填充大小是kernel-1
        self.trd = TrendRepresentationDisentangler(args)

        # projection head
        self.head = nn.Sequential(
            nn.Linear(args.repr_dims // 2, args.repr_dims // 2),
            nn.ReLU(),
            nn.Linear(args.repr_dims // 2, args.repr_dims // 2)
        )

        # seasonal representation Disentangler使用多频带傅里叶层网络结构
        self.srd = MultiBandSeasonalDisentangler(
            in_channels=args.repr_dims,
            out_channels=args.repr_dims // 2,
            num_bands=3,  # 日、周、年三个频带
            length=args.seq_len
        )

        self.repr_dropout = nn.Dropout(p=0.1)
        # 趋势/季节性表示特征归一化层：解决两种表示特征量值差异大的问题
        self.trend_norm = nn.LayerNorm(args.repr_dims // 2)
        self.season_norm = nn.LayerNorm(args.repr_dims // 2)

        # 耦合门控单元
        self.cgu = CoupledGatingUnit(args.repr_dims // 2)

        self.feature_reducer= FeatureReducer(args,args.hidden_dims)#将解耦表示的维度从repr_dims  映射回到 hidden_dims
        
        self.projection = nn.Linear(args.hidden_dims, args.c_out, bias=True)  # 新增的全连接层
        self.pool = nn.AdaptiveAvgPool1d(output_size=args.pred_len )  # 自适应平均池化层

        if device == 'cuda:{}'.format(self.args.gpu):
            self._move_to_cuda()


    
    def _move_to_cuda(self):
        """将组件移动到CUDA设备"""
        components = [
            self.feature_extractor, self.trd, self.head, self.srd,
            self.feature_reducer, self.projection, self.pool
        ]
        for component in components:
            component = component.cuda()
        if self.input_projection_list is not None:
            self.input_projection_list = self.input_projection_list.cuda()
    
    def _init_input_projections(self, num_vars, hidden_dim):
        """动态初始化输入投影层,为每个变量进行独立地映射"""
        self.input_projection_list = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_vars)
        ]).to(self.device)
        print(f"动态初始化输入投影层: {num_vars} 个变量 -> 隐藏维度 {hidden_dim}")
  
    def _freeze_pretrained_components(self, transferred_layers=None):
        """智能冻结策略"""
        # 总是冻结这些组件
        always_freeze = [self.trd, self.head, self.srd]
        for component in always_freeze:
            for param in component.parameters():
                param.requires_grad = False
        
        # 对特征提取器进行智能冻结
        if self.feature_extractor is not None and transferred_layers:
            for layer_info in transferred_layers:
                if layer_info['type'] == 'group_conv':
                    # 冻结迁移的分组卷积层
                    layer_idx = layer_info['layer_idx']
                    block_idx = layer_info['block_idx']
                    component_name = layer_info['component']
                    
                    if (layer_idx < len(self.feature_extractor.group_convs) and
                        block_idx < len(self.feature_extractor.group_convs[layer_idx])):
                        
                        block = self.feature_extractor.group_convs[layer_idx][block_idx]
                        if component_name == 'conv1' and hasattr(block, 'conv1'):
                            for param in block.conv1.parameters():
                                param.requires_grad = False
                        elif component_name == 'conv2' and hasattr(block, 'conv2'):
                            for param in block.conv2.parameters():
                                param.requires_grad = False
                                
                elif layer_info['type'] == 'final_conv':
                    # 冻结迁移的final_conv
                    component_name = layer_info['component']
                    if component_name == 'weight':
                        self.feature_extractor.final_conv.weight.requires_grad = False
                    elif component_name == 'bias':
                        self.feature_extractor.final_conv.bias.requires_grad = False
    
        print("冻结预训练组件完成")

    def _freeze_transferred_layers(self, transferred_layers):
        """只冻结成功迁移的层"""
        for layer_info in transferred_layers:
            if layer_info['type'] == 'group_conv':
                # 冻结迁移的分组卷积层
                group_idx = layer_info['group_idx']
                layer_idx = layer_info['layer_idx']
                block_idx = layer_info['block_idx']
                
                if (group_idx < len(self.feature_extractor.group_convs) and
                    layer_idx < len(self.feature_extractor.group_convs[group_idx]) and
                    block_idx < len(self.feature_extractor.group_convs[group_idx][layer_idx])):
                    
                    block = self.feature_extractor.group_convs[group_idx][layer_idx][block_idx]
                    for param in block.parameters():
                        param.requires_grad = False
                        
            elif layer_info['type'] == 'final_conv':
                # 冻结迁移的final_conv
                for param in self.feature_extractor.final_conv.parameters():
                    param.requires_grad = False
  
    def _transfer_encoder_weights(self, pretrained_state_dict, tgt_encoder):
        """从state_dict迁移权重到目标编码器"""
        transferred_layers = []
        
        # 处理可能的多GPU前缀
        state_dict = {}
        for k, v in pretrained_state_dict.items():
            if k.startswith('module.'):
                # 移除多GPU前缀
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        
        # 迁移分组卷积权重
        for layer_idx in range(len(tgt_encoder.group_convs)):
            tgt_group_conv = tgt_encoder.group_convs[layer_idx]
            
            # 只迁移深层
            # 浅层网络（layer_idx < freeze_start_layer）重新初始化，不进行权重迁移，保持可训练状态
            if layer_idx >= getattr(self.args, 'freeze_start_layer', 2):
                for block_idx in range(len(tgt_group_conv)):
                    block = tgt_group_conv[block_idx]
                    
                    # 构建状态字典中的键名
                    conv1_key = f'feature_extractor.group_convs.{layer_idx}.{block_idx}.conv1.conv.weight'
                    conv2_key = f'feature_extractor.group_convs.{layer_idx}.{block_idx}.conv2.conv.weight'
                    
                    # 迁移conv1
                    if conv1_key in state_dict and hasattr(block, 'conv1'):
                        src_weight = state_dict[conv1_key]
                        if src_weight.shape == block.conv1.conv.weight.shape: # 检查维度匹配性
                            block.conv1.conv.weight.data = src_weight.clone() # 成功匹配则迁移权重并记录
                            transferred_layers.append({
                                'type': 'group_conv',
                                'layer_idx': layer_idx,
                                'block_idx': block_idx,
                                'component': 'conv1'
                            })
                    
                    # 迁移conv2
                    if conv2_key in state_dict and hasattr(block, 'conv2'):
                        src_weight = state_dict[conv2_key]
                        if src_weight.shape == block.conv2.conv.weight.shape:
                            block.conv2.conv.weight.data = src_weight.clone()
                            transferred_layers.append({
                                'type': 'group_conv', 
                                'layer_idx': layer_idx,
                                'block_idx': block_idx,
                                'component': 'conv2'
                            })
        
        # 迁移final_conv权重
        final_conv_weight_key = 'feature_extractor.final_conv.weight'
        final_conv_bias_key = 'feature_extractor.final_conv.bias'
        
        if (final_conv_weight_key in state_dict and #检查维度兼容性和在目标目标模型中的存在性
            state_dict[final_conv_weight_key].shape == tgt_encoder.final_conv.weight.shape):
            
            tgt_encoder.final_conv.weight.data = state_dict[final_conv_weight_key].clone()
            transferred_layers.append({'type': 'final_conv', 'component': 'weight'})
        
        if (final_conv_bias_key in state_dict and 
            state_dict[final_conv_bias_key].shape == tgt_encoder.final_conv.bias.shape):
            
            tgt_encoder.final_conv.bias.data = state_dict[final_conv_bias_key].clone()
            transferred_layers.append({'type': 'final_conv', 'component': 'bias'})
        
        return tgt_encoder, transferred_layers

    def _transfer_conv_block_weights(self, src_block, tgt_block):
        """迁移卷积块权重，返回是否成功迁移"""
        transferred = False
        
        # 迁移conv1
        if (hasattr(src_block, 'conv1') and hasattr(tgt_block, 'conv1') and
            src_block.conv1.conv.weight.shape == tgt_block.conv1.conv.weight.shape):
            
            tgt_block.conv1.conv.weight.data = src_block.conv1.conv.weight.data.clone()
            if src_block.conv1.conv.bias is not None:
                tgt_block.conv1.conv.bias.data = src_block.conv1.conv.bias.data.clone()
            transferred = True
        
        # 迁移conv2  
        if (hasattr(src_block, 'conv2') and hasattr(tgt_block, 'conv2') and
            src_block.conv2.conv.weight.shape == tgt_block.conv2.conv.weight.shape):
            
            tgt_block.conv2.conv.weight.data = src_block.conv2.conv.weight.data.clone()
            if src_block.conv2.conv.bias is not None:
                tgt_block.conv2.conv.bias.data = src_block.conv2.conv.bias.data.clone()
            transferred = True
        
        return transferred

    def forward(self, x,groups,pretrained_state_dict=None):# 输入的x的形状为 b,seq_len,input_size
        """前向传播"""
        
        batch_size, seq_len, input_size = x.shape
        
        # 初始化特征提取器
        if self.feature_extractor is None:
            self.feature_extractor = MixedChannelConvEncoder(
                self.args.hidden_dims,
                self.args.repr_dims, 
                self.args.kernel_size,
                groups,
                self.args.depth,
                self.device
            ).to(self.device)
        
        # 如果有预训练权重，进行迁移
        transferred_layers = []
        if pretrained_state_dict is not None:
            self.feature_extractor, transferred_layers = self._transfer_encoder_weights(
                pretrained_state_dict, self.feature_extractor
            )

        # 在微调阶段执行智能冻结
        if self.stage2 and not hasattr(self, '_frozen'):
            self._freeze_pretrained_components(transferred_layers)
            self._frozen = True
        
        # 应用输入投影
        projected_vars = []
        for var_idx in range(input_size):
            var_data = x[:, :, var_idx:var_idx+1]  # [batch, seq_len, 1]
            projected_var = self.input_projection_list[var_idx](var_data)  # [batch, seq_len, hidden_dim]
            projected_vars.append(projected_var)
        # 拼接所有变量 [batch, seq_len, input_size * hidden_dims]
        x_projected = torch.cat(projected_vars, dim=-1)
        y = self.feature_extractor(x_projected) # [batch, repr_dims,seq_len]
    
        #提去趋势性成分特征
        trend = []
        for idx, mod in enumerate(self.trd.conv_layers):
            out = mod(y.to(self.device))
            if self.args.kernels[idx] != 1:
                out = out[..., :-(self.args.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))
        trend = torch.mean(torch.stack(trend), dim=0)
        trend = self.trend_norm(trend)  # 归一化
        
        # 提取季节性成分的特征
        season = self.srd(y.transpose(1, 2)) # [B, T, repr_dims//2]
        season = self.season_norm(season)  # 归一化
        
        # 使用CGU进行双向调制
        trend_modulated, season_modulated = self.cgu(trend, season)

        # 趋势/季节特征均应用dropout
        trend = self.repr_dropout(trend_modulated)  
        season = self.repr_dropout(season_modulated)

        if not self.stage2:
            return trend, season
        else:
            output = torch.cat([trend, season], dim=2) # (batch_size,seq_len,repr_dims)
            output = (self.feature_reducer(output.transpose(1, 2))).transpose(1,2) # b  t input_size 

            output = self.projection(output)  # 应用全连接层，使维度从 (batch_size, t, input_size) 转换为 (batch_size, t, 1)
            output = self.pool(output.transpose(1, 2)).transpose(1, 2)  # 使用自适应平均池化调整时间步数
            return output[:, -self.args.pred_len:, :]    

    def compute_loss(self, batch_x, batch_y, plot_dir, groups, plot_augment_flag):
        batch_x, positive_batch_x, negative_batch_x_list = augment(
            batch_x, batch_y, self.args.num_augment, plot_dir, 
            self.args.plot_augment, plot_augment_flag
        )
        
        batch_x = batch_x.to(torch.float32)
        batch_x = batch_x.transpose(1, 2).to(self.device)
        
        # 获取原样本表示
        output_t, output_s = self.forward(batch_x, groups)  # [B, seq_len, repr_dims//2]
        
        # 获取正样本表示
        positive_batch_x = positive_batch_x.transpose(1, 2)
        output_positive_t, output_positive_s = self.forward(positive_batch_x.float().to(self.device), groups)
        
        # 获取负样本表示
        output_negative_t_list = []
        output_negative_s_list = []
        for negative_batch_x in negative_batch_x_list:
            negative_batch_x = negative_batch_x.to(torch.float32)
            negative_batch_x = negative_batch_x.transpose(1, 2)
            output_negative_t, output_negative_s = self.forward(negative_batch_x.float().to(self.device), groups)
            output_negative_t_list.append(output_negative_t)
            output_negative_s_list.append(output_negative_s)
        
        # 改进的趋势对比损失计算
        trend_loss = self._compute_trend_contrastive_loss(output_t, output_positive_t, output_negative_t_list)
        
        # 修正的季节性对比损失计算（时域）
        seasonal_loss = self._compute_seasonal_contrastive_loss(output_s, output_positive_s, output_negative_s_list)
        
        # 计算总对比损失
        loss = trend_loss + self.args.alpha * seasonal_loss
        return loss

    def _compute_trend_contrastive_loss(self, anchor_t, pos_t, neg_t_list):
        """趋势对比损失 - 多时间步加权"""
        B, seq_len, C = anchor_t.shape
        
        # 选择关键时间点（避免随机性）
        key_indices = [0, seq_len//4, seq_len//2, 3*seq_len//4, -1]  # 均匀采样关键点
        if len(key_indices) > seq_len:
            key_indices = list(range(seq_len))
        
        losses = []
        for idx in key_indices:
            # 处理锚点
            anchor_feat = anchor_t[:, idx, :]  # [B, C]
            anchor_feat = F.normalize(self.head(anchor_feat), dim=-1)
            
            # 处理正样本
            pos_feat = pos_t[:, idx, :]
            pos_feat = F.normalize(self.head(pos_feat), dim=-1)
            
            # 处理负样本
            neg_feats = []
            for neg_t in neg_t_list:
                neg_feat = neg_t[:, idx, :]
                neg_feat = F.normalize(self.head(neg_feat), dim=-1)
                neg_feats.append(neg_feat)
            neg_feats_all = torch.stack(neg_feats, dim=1)  # [B, λ, C]
            
            # 计算该时间点的对比损失
            point_loss = self.caculate_unified_contrastive_loss(
                anchor_feat, pos_feat, neg_feats_all
            )
            losses.append(point_loss)
        
        # 对多个时间点的损失进行平均
        return torch.mean(torch.stack(losses))

    def _compute_seasonal_contrastive_loss(self, anchor_s, pos_s, neg_s_list):
        """季节性对比损失 - 频域多分量对比，利用负样本"""
        B, seq_len, C = anchor_s.shape
        
        # 转换为频域
        anchor_freq = fft.rfft(anchor_s, dim=1)  # [B, freq_bins, C]
        pos_freq = fft.rfft(pos_s, dim=1)
        
        # 获取幅度和相位
        anchor_amp, anchor_phase = self.convert_coeff(anchor_freq)
        pos_amp, pos_phase = self.convert_coeff(pos_freq)
        
        # 选择关键频率分量
        freq_bins = anchor_amp.shape[1]
        key_freq_indices = [0, freq_bins//4, freq_bins//2, -1]  # 低频、中频、高频
        
        losses = []
        for freq_idx in key_freq_indices:
            if freq_idx >= freq_bins:
                continue
                
            # 幅度对比
            anchor_amp_feat = anchor_amp[:, freq_idx, :]  # [B, C]
            pos_amp_feat = pos_amp[:, freq_idx, :]
            
            # 处理负样本的幅度
            neg_amp_feats = []
            for neg_s in neg_s_list:
                neg_freq = fft.rfft(neg_s, dim=1)
                neg_amp, _ = self.convert_coeff(neg_freq)
                neg_amp_feat = neg_amp[:, freq_idx, :]
                neg_amp_feats.append(neg_amp_feat)
            neg_amp_feats_all = torch.stack(neg_amp_feats, dim=1)  # [B, λ, C]
            
            amp_loss = self.caculate_unified_contrastive_loss(
                anchor_amp_feat, pos_amp_feat, neg_amp_feats_all
            )
            
            # 相位对比
            anchor_phase_feat = anchor_phase[:, freq_idx, :]
            pos_phase_feat = pos_phase[:, freq_idx, :]
            
            neg_phase_feats = []
            for neg_s in neg_s_list:
                neg_freq = fft.rfft(neg_s, dim=1)
                _, neg_phase = self.convert_coeff(neg_freq)
                neg_phase_feat = neg_phase[:, freq_idx, :]
                neg_phase_feats.append(neg_phase_feat)
            neg_phase_feats_all = torch.stack(neg_phase_feats, dim=1)
            
            phase_loss = self.caculate_unified_contrastive_loss(
                anchor_phase_feat, pos_phase_feat, neg_phase_feats_all
            )
            
            losses.append(amp_loss + phase_loss)
        
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0)


    def caculate_unified_contrastive_loss(self, anchor, pos, negs, temperature=None):
        """统一的对比损失函数，适用于趋势和季节性成分"""
        if temperature is None:
            temperature = self.args.temperature
        
        # 归一化所有特征
        anchor = F.normalize(anchor, dim=-1)
        pos = F.normalize(pos, dim=-1) 
        negs = F.normalize(negs, dim=-1)
        
        # 计算相似度
        pos_similarity = torch.sum(anchor * pos, dim=-1, keepdim=True)  # [B, 1]
        neg_similarity = torch.sum(anchor.unsqueeze(1) * negs, dim=-1)  # [B, λ]
        
        # 合并logits
        logits = torch.cat([pos_similarity, neg_similarity], dim=-1)  # [B, 1+λ]
        
        # 应用温度参数
        logits = logits / temperature
        
        # 创建标签（正样本在位置0）
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss 

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase
