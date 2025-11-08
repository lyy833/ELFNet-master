import torch 
from torch import nn
import torch.nn.functional as F
from layers.dilated_conv import DilatedConvEncoder,ConvBlock
import torch.fft as fft
import math
from utils.augmentation import augment
import numpy as np
import warnings

warnings.filterwarnings('ignore')



class MixedChannelConvEncoder(nn.Module):
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


class ELFNet(nn.Module):
    def __init__(self,args,  device, stage2=False):
        super(ELFNet, self).__init__()

        self.args = args
        self.stage2 = stage2
        self.device = device

        # 动态初始化的输入投影层列表
        self.input_projection_list = None

        ### 变量分组混合通道
        self.feature_extractor = None # 变量分组可能发生变化，前向传播中在具体确定
        
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

        self.feature_reducer= FeatureReducer(args,args.hidden_dims)#将解耦表示的维度从repr_dims  映射回到 hidden_dims
        
        self.projection = nn.Linear(args.hidden_dims, args.c_out, bias=True)  # 新增的全连接层
        self.pool = nn.AdaptiveAvgPool1d(output_size=args.pred_len )  # 自适应平均池化层

        if device == 'cuda:{}'.format(self.args.gpu):
            self._move_to_cuda()


    
    def _move_to_cuda(self):
        """将组件移动到CUDA设备"""
        components = [
            self.feature_extractor, self.tfd, self.head, self.sfd,
            self.feature_reducer, self.projection, self.pool
        ]
        for component in components:
            component = component.cuda()
        if self.input_projection_list is not None:
            self.input_projection_list = self.input_projection_list.cuda()
    
    def _freeze_pretrained_components(self, transferred_layers=None):
        """智能冻结策略 - 只冻结迁移过来的层"""
        # 总是冻结这些组件
        always_freeze = [self.tfd, self.head, self.sfd]
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

    def _init_input_projections(self, num_vars, hidden_dim):
        """动态初始化输入投影层,为每个变量进行独立地映射"""
        if self.input_projection_list is None:
            self.input_projection_list = nn.ModuleList([
                nn.Linear(1, hidden_dim) for _ in range(num_vars)
            ]).to(self.device)
            print(f"动态初始化输入投影层: {num_vars} 个变量 -> 隐藏维度 {hidden_dim}")
    
    
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
            if layer_idx >= getattr(self.args, 'freeze_start_layer', 2):
                for block_idx in range(len(tgt_group_conv)):
                    block = tgt_group_conv[block_idx]
                    
                    # 构建状态字典中的键名
                    conv1_key = f'feature_extractor.group_convs.{layer_idx}.{block_idx}.conv1.conv.weight'
                    conv2_key = f'feature_extractor.group_convs.{layer_idx}.{block_idx}.conv2.conv.weight'
                    
                    # 迁移conv1
                    if conv1_key in state_dict and hasattr(block, 'conv1'):
                        src_weight = state_dict[conv1_key]
                        if src_weight.shape == block.conv1.conv.weight.shape:
                            block.conv1.conv.weight.data = src_weight.clone()
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
        
        if (final_conv_weight_key in state_dict and 
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
        
        # 初始化输入投影层和特征提取器
        self._init_input_projections(input_size, self.args.hidden_dims)
        
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
    
        #总的来说，下面代码片段的目标是捕获输入数据x在不同时间尺度上的趋势，并计算这些趋势的平均值。使用多尺度的方法是时间序列分析中的一种常见技巧，因为它允许模型在不同的时间尺度上捕获模式和依赖性。
        #提去趋势性成分特征
        trend = []
        for idx, mod in enumerate(self.tfd.conv_layers):
            out = mod(y.to(self.device))
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
    


    def compute_loss(self,batch_x,batch_y,plot_dir,groups):
        batch_x, positive_batch_x, negative_batch_x_list= augment(batch_x,batch_y,self.args.num_augment,plot_dir,self.args.plot)
        
        rand_idx = np.random.randint(0, batch_x.shape[1]) 
        
        # 计算原样本的趋势性输出
        batch_x = batch_x.to(torch.float32)
        batch_x = batch_x.transpose(1, 2).to(self.device)
        output_t, output_s= self.forward(batch_x,groups) # (b,seq_len,repr_dims/2) 
        if output_t is not None:
            output_t = F.normalize(self.head(output_t[:, rand_idx]), dim=-1)
        
        # 计算正样本的趋势性输出
        #positive_batch_x = torch.from_numpy(positive_batch_x.astype('float32'))
        positive_batch_x = positive_batch_x.transpose(1, 2)
        output_positive_t, output_positive_s= self.forward(positive_batch_x.float().to(self.device),groups) # (b,seq_len,repr_dims/2) 
        if output_positive_t is not None:
            output_positive_t = F.normalize(self.head(output_positive_t[:, rand_idx]), dim=-1) #(batch_size,seq_len,repr_dims/2)
        
        # 计算所有负样本的趋势性输出
        output_negative_t_list = []
        for negative_batch_x in negative_batch_x_list:
            # 计算每个负样本的趋势性输出
            negative_batch_x = negative_batch_x.to(torch.float32)
            negative_batch_x = negative_batch_x.transpose(1, 2)
            output_negative_t, _= self.forward(negative_batch_x.float().to(self.device),groups) # (b,seq_len,repr_dims/2) 
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
    

    