import torch 
from torch import nn
import torch.nn.functional as F
from models.ELFNet import TrendRepresentationDisentangler, MultiBandSeasonalDisentangler,FeatureReducer,MixedChannelConvEncoder,CoupledGatingUnit,BandedFourierLayer
import torch.fft as fft
import warnings
import torch
from models.DLinear import series_decomp  # 从 DLinear 导入滑动平均分解类
from models.ELFNet import MixedChannelConvEncoder,FeatureReducer
from utils.augmentation import VariableImportanceAnalyzer,TemporalAugmenter,TrendSeasonalAugmenter,CausalAwareAugmenter,DynamicPeakDetectionAugmenter,DomainAugmentationFramework
warnings.filterwarnings('ignore')

   
class CommonAugmenter:
    """
    普通时序增强器。
    """
    def __init__(self):
        pass

    def jitter(self, x, sigma=0.03):
        """抖动：添加高斯噪声"""
        return x + torch.randn_like(x) * sigma

    def scaling(self, x, sigma=0.1):
        """缩放：每个样本-通道乘以随机因子"""
        factor = torch.randn(x.size(0), x.size(2), device=x.device) * sigma + 1.0
        return x * factor.unsqueeze(1)  # [B, 1, C] 广播到 [B, T, C]

    def shift(self, x, max_shift=10):
        """时间轴平移：循环移位（模拟事件位置扰动）"""
        shift = torch.randint(-max_shift, max_shift, (1,)).item()
        return torch.roll(x, shifts=shift, dims=1)

    def reverse_order(self, x):
        """逆序：沿时间轴反转"""
        return torch.flip(x, dims=[1])

    def detrend(self, x):
        """去趋势：减去线性拟合的趋势（简易版）"""
        T = x.size(1)
        t = torch.linspace(0, 1, T, device=x.device).view(1, T, 1)  # [1, T, 1]
        # 最小二乘拟合趋势
        x_mean = x.mean(dim=1, keepdim=True)
        t_mean = t.mean(dim=1, keepdim=True)
        slope = ((x - x_mean) * (t - t_mean)).sum(dim=1, keepdim=True) / ((t - t_mean)**2).sum(dim=1, keepdim=True)
        trend = slope * t
        return x - trend

    def cumulative_sum(self, x):
        """累积和（破坏性操作，通常不适合作为增强）"""
        return torch.cumsum(x, dim=1)

    def polynomial_transform(self, x, degree=2):
        """多项式变换（破坏性操作）"""
        return torch.pow(x, degree)
    def augment(self,x):
        # 正增强
        x_jitter = self.jitter(x)
        x_scaling = self.scaling(x_jitter)
        x_shift = self.shift(x_scaling)
        x_augment_p = x_shift  

        # 负增强
        x_augment_n_list = []
        x_augment_n1 = self.reverse_order(x)
        x_augment_n_list.append(x_augment_n1)
        x_augment_n2 = self.detrend(x)
        x_augment_n_list.append(x_augment_n2)
        x_augment_n3 = self.cumulative_sum(x)
        x_augment_n_list.append(x_augment_n3)
        x_augment_n4 = self.polynomial_transform(x)
        x_augment_n_list.append(x_augment_n4)

        return x,x_augment_p,x_augment_n_list

class DomainAugmentationFramework_ablation:
    """
    基于领域知识的数据增强框架
    """
    def __init__(self, args,wo_augmentor):
        self.target_index = args.pretrain_target_idx
        self.importance_analyzer = VariableImportanceAnalyzer(args.pretrain_target_idx)
        self.causal_augmenter = None
        self.peak_augmenter = DynamicPeakDetectionAugmenter()
        self.temporal_augmenter = TemporalAugmenter()
        self.trend_seasonal_augmenter = TrendSeasonalAugmenter(args.pretrain_freq)
        self.wo_augmentor = wo_augmentor
        self.common_augmentor = CommonAugmenter()

        
    def initialize_from_data(self, data_x, data_y):
        """初始化因果感知增强器"""
        importance_scores = self.importance_analyzer.analyze_variable_importance(data_x, data_y)
        causal_vars, non_causal_vars = self.importance_analyzer.select_causal_variables(importance_scores)
        self.causal_augmenter = CausalAwareAugmenter(causal_vars, non_causal_vars)
        return causal_vars, non_causal_vars
    
    def augment_batch(self, batch_x):
        """批处理增强 - 各司其职"""
        if self.wo_augmentor == 'CausalAwareAugmenter':
            positive_x = self.peak_augmenter.dynamic_peak_augment(batch_x, self.target_index)
            trend_negative_list = [self.trend_seasonal_augmenter.corrupt_for_trend(positive_x)]
            seasonal_negative_list = [self.trend_seasonal_augmenter.corrupt_for_seasonality(positive_x)]
            negative_x_list = []

        elif self.wo_augmentor == 'DynamicPeakDetectionAugmenter':
            batch_x, positive_x, negative_x_list,causal_vars,non_causal_vars = self.causal_augmenter.augment_batch(batch_x)
            trend_negative_list = [self.trend_seasonal_augmenter.corrupt_for_trend(positive_x)]
            seasonal_negative_list = [self.trend_seasonal_augmenter.corrupt_for_seasonality(positive_x)]
        
        elif self.wo_augmentor == 'TrendSeasonalAugmenter':
            batch_x, positive_x, negative_x_list,causal_vars,non_causal_vars = self.causal_augmenter.augment_batch(batch_x)
            positive_x = self.peak_augmenter.dynamic_peak_augment(batch_x, self.target_index)
            trend_negative_list = []
            seasonal_negative_list = []
        else:
            batch_x, positive_x, negative_x_list = self.common_augmentor.augment(batch_x)
            trend_negative_list = []
            seasonal_negative_list = []

        # 通用增强：丰富正负样本
        ## 负样本处理
        if negative_x_list != []:
            negative_x_list = [self.temporal_augmenter._fluctuation_pattern_augment(x)
                           for x in negative_x_list]
        else:
            negative_x_list = []
        
        if trend_negative_list != []:
            t_negative_list = [self.temporal_augmenter._fluctuation_pattern_augment(x)
                           for x in trend_negative_list]
        else:
            t_negative_list = []

        if seasonal_negative_list != []:
            s_negative_list = [self.temporal_augmenter._fluctuation_pattern_augment(x)
                           for x in seasonal_negative_list]
        else:
            s_negative_list = []

        ## 正样本处理
        positive_x = self.temporal_augmenter._fluctuation_pattern_augment(positive_x)
        
        return batch_x, positive_x, negative_x_list,t_negative_list,s_negative_list
        
    

class ELFNet_wo_TS(nn.Module):
    def __init__(self,args, device):
        """wo_
        有监督版ELFNet
        """
        super(ELFNet_wo_TS, self).__init__()

        self.args = args
        self.device = device
        self.stage = 1 # 单阶段，直接写死

        # 动态初始化的输入投影层列表和特征提取器
        self.input_projection_list = None
        self.feature_extractor = None
        
        self.feature_reducer= FeatureReducer(args,args.hidden_dims)#将解耦表示的维度从repr_dims  映射回到 input_size
        
        self.projection = nn.Linear(args.hidden_dims, args.c_out, bias=True)  # 新增的全连接层
        self.pool = nn.AdaptiveAvgPool1d(output_size=args.pred_len )  # 自适应平均池化层


    def _init_input_projections(self, num_vars, hidden_dim):
        """动态初始化输入投影层,为每个变量进行独立地映射"""
        self.input_projection_list = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_vars)
        ]).to(self.device)
        print(f"动态初始化输入投影层: {num_vars} 个变量 -> 隐藏维度 {hidden_dim} (move to {self.device})")
    
    def _init_feature_extractor(self,groups):
        self.feature_extractor = MixedChannelConvEncoder(
                self.args.hidden_dims,
                self.args.repr_dims,
                self.args.kernel_size,
                groups,
                self.args.depth
            ).to(self.device)
        print(f"动态初始化特征提取器, 变量分组: {groups} (move to {self.device})")


    def forward(self, x,pretrained_state_dict=None):# 输入的x的形状为 b,input_size,seq_len
        batch_size, seq_len, input_size = x.shape
            
        # 应用输入投影
        projected_vars = []
        for var_idx in range(input_size):
            var_data = x[:, :, var_idx:var_idx+1]  # [batch, seq_len, 1]
            projected_var = self.input_projection_list[var_idx](var_data)  # [batch, seq_len, hidden_dim]
            projected_vars.append(projected_var)
        # 拼接所有变量 [batch, seq_len, input_size * hidden_dim]
        x_projected = torch.cat(projected_vars, dim=-1)
        y = self.feature_extractor(x_projected) # [batch, repr_dims,seq_len]

        output = y.transpose(1,2)# (batch_size,seq_len,repr_dims)
        output = (self.feature_reducer(output.transpose(1, 2))).transpose(1,2) # b  t input_size 

        output = self.projection(output)  # 应用全连接层，使维度从 (batch_size, t, input_size) 转换为 (batch_size, t, 1)
        output = self.pool(output.transpose(1, 2)).transpose(1, 2)  # 使用自适应平均池化调整时间步数
        return output[:, -self.args.pred_len:, :]    


class ELFNet_ablation_augmentor(nn.Module):
    def __init__(self,args, wo_augmentor, device, stage=1):
        """
        自监督版ELFNet
        """
        super(ELFNet_ablation_augmentor, self).__init__()

        self.args = args
        self.stage = stage # 1表示预训练阶段，2表示微调阶段，3表示测试阶段
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
        
        self.augmentor = DomainAugmentationFramework_ablation(args,wo_augmentor)
        # one-time transfer flag
        self._is_transferred = False
        

    def _init_input_projections(self, num_vars, hidden_dim):
        """动态初始化输入投影层,为每个变量进行独立地映射"""
        self.input_projection_list = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_vars)
        ]).to(self.device)
        print(f"动态初始化输入投影层: {num_vars} 个变量 -> 隐藏维度 {hidden_dim} (move to {self.device})")
    
    def _init_feature_extractor(self,groups):
        # 初始化特征提取器
        self.feature_extractor = MixedChannelConvEncoder(
            self.args.hidden_dims,
            self.args.repr_dims, 
            self.args.kernel_size,
            groups,
            self.args.depth,
        ).to(self.device)
        print(f"动态初始化特征提取器, 变量分组: {groups} (move to {self.device})")

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
                    if conv1_key in pretrained_state_dict and hasattr(block, 'conv1'):
                        src_weight = pretrained_state_dict[conv1_key]
                        if src_weight.shape == block.conv1.conv.weight.shape: # 检查维度匹配性
                            block.conv1.conv.weight.data = src_weight.clone() # 成功匹配则迁移权重并记录
                            transferred_layers.append({
                                'type': 'group_conv',
                                'layer_idx': layer_idx,
                                'block_idx': block_idx,
                                'component': 'conv1'
                            })
                    
                    # 迁移conv2
                    if conv2_key in pretrained_state_dict and hasattr(block, 'conv2'):
                        src_weight = pretrained_state_dict[conv2_key]
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
        
        if (final_conv_weight_key in pretrained_state_dict and #检查维度兼容性和在目标目标模型中的存在性
            pretrained_state_dict[final_conv_weight_key].shape == tgt_encoder.final_conv.weight.shape):
            
            tgt_encoder.final_conv.weight.data = pretrained_state_dict[final_conv_weight_key].clone()
            transferred_layers.append({'type': 'final_conv', 'component': 'weight'})
        
        if (final_conv_bias_key in pretrained_state_dict and 
            pretrained_state_dict[final_conv_bias_key].shape == tgt_encoder.final_conv.bias.shape):
            
            tgt_encoder.final_conv.bias.data = pretrained_state_dict[final_conv_bias_key].clone()
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

    def forward(self, x,pretrained_state_dict=None):# 输入的x的形状为 b,seq_len,input_size
        """前向传播"""
        
        batch_size, seq_len, input_size = x.shape
        
        # 在微调阶段，如果有预训练权重，进行迁移（只迁移一次，避免在每次 forward 都重复复制权重）
        transferred_layers = []
        if pretrained_state_dict is not None and not getattr(self, '_is_transferred') and self.stage==2:
            self.feature_extractor, transferred_layers = self._transfer_encoder_weights(
                pretrained_state_dict, self.feature_extractor
            )
            # 设置标志以避免重复迁移
            self._is_transferred = True

        # 在微调阶段执行智能冻结
        if self.stage==2 and not hasattr(self, '_frozen'):
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
    
        #提取趋势性成分特征
        trend = []
        for idx, mod in enumerate(self.trd.conv_layers):
            out = mod(y)
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

        if self.stage==1: # 预训练阶段使用自监督表示学习，输出两种成分
            return trend, season
        else:
            output = torch.cat([trend, season], dim=2) # (batch_size,seq_len,repr_dims)
            output = (self.feature_reducer(output.transpose(1, 2))).transpose(1,2) # b  t input_size 

            output = self.projection(output)  # 应用全连接层，使维度从 (batch_size, t, input_size) 转换为 (batch_size, t, 1)
            output = self.pool(output.transpose(1, 2)).transpose(1, 2)  # 使用自适应平均池化调整时间步数
            return output[:, -self.args.pred_len:, :]    

    def compute_loss(self, batch_x, plot_dir, plot_augment_flag):
        if not self.args.plot_augment:
            plot_dir = None
        batch_x, positive_batch_x, negative_batch_x_list,trend_negative_batch_x_list,seasonal_negative_batch_x_list = self.augmentor.augment_batch(batch_x,plot_dir, plot_augment_flag)
        t_negative_batch_x_list = negative_batch_x_list + trend_negative_batch_x_list
        s_negative_batch_x_list = negative_batch_x_list + seasonal_negative_batch_x_list

        batch_x = batch_x.float().transpose(1, 2)

        # 获取原样本表示
        output_t, output_s = self.forward(batch_x)  # [B, seq_len, repr_dims//2]
        
        # 获取正样本表示
        positive_batch_x = positive_batch_x.transpose(1, 2).float()
        output_positive_t, output_positive_s = self.forward(positive_batch_x)
        
        # 分别获取趋势性/季节性负样本表示
        output_negative_t_list = []
        for negative_batch_x in t_negative_batch_x_list:
            negative_batch_x = negative_batch_x.transpose(1, 2).float()
            output_negative_t, _= self.forward(negative_batch_x)
            output_negative_t_list.append(output_negative_t)
        output_negative_s_list = []
        for negative_batch_x in s_negative_batch_x_list:
            negative_batch_x = negative_batch_x.transpose(1, 2).float()
            _, output_negative_s = self.forward(negative_batch_x)
            output_negative_s_list.append(output_negative_s)
        
        # 趋势性成分对比损失计算
        trend_loss = self._compute_trend_contrastive_loss(output_t, output_positive_t, output_negative_t_list)
        
        # 季节性成分对比损失计算（时域）
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


class ELFNet_supervised_pretrain(nn.Module):
    """
    有监督预训练版ELFNet。
    与自监督版ELFNet一样采用两阶段训练：
      - Stage 1: 有监督预训练（使用MSE损失，利用batch_y标签）
      - Stage 2: 微调（与自监督版相同，加载stage1权重后进行权重迁移与冻结）
    与ELFNet_supervised（单阶段有监督）的核心区别在于采用两阶段训练策略。
    """
    def __init__(self, args, device, stage=1):
        super(ELFNet_supervised_pretrain, self).__init__()

        self.args = args
        self.stage = stage
        self.device = device

        # 动态初始化的输入投影层列表和特征提取器
        self.input_projection_list = None
        self.feature_extractor = None

        # Trend Representation Disentangler
        self.trd = TrendRepresentationDisentangler(args)

        # Seasonal Representation Disentangler（多频带傅里叶层）
        self.srd = MultiBandSeasonalDisentangler(
            in_channels=args.repr_dims,
            out_channels=args.repr_dims // 2,
            num_bands=3,
            length=args.seq_len
        )

        self.repr_dropout = nn.Dropout(p=0.1)
        # 趋势/季节性表示特征归一化层
        self.trend_norm = nn.LayerNorm(args.repr_dims // 2)
        self.season_norm = nn.LayerNorm(args.repr_dims // 2)

        # 耦合门控单元
        self.cgu = CoupledGatingUnit(args.repr_dims // 2)

        # 特征还原与预测头
        self.feature_reducer = FeatureReducer(args, args.hidden_dims)
        self.projection = nn.Linear(args.hidden_dims, args.c_out, bias=True)
        self.pool = nn.AdaptiveAvgPool1d(output_size=args.pred_len)

        # one-time transfer flag
        self._is_transferred = False

    def _init_input_projections(self, num_vars, hidden_dim):
        """动态初始化输入投影层，为每个变量进行独立映射"""
        self.input_projection_list = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_vars)
        ]).to(self.device)
        print(f"动态初始化输入投影层: {num_vars} 个变量 -> 隐藏维度 {hidden_dim} (move to {self.device})")

    def _init_feature_extractor(self, groups):
        """动态初始化特征提取器"""
        self.feature_extractor = MixedChannelConvEncoder(
            self.args.hidden_dims,
            self.args.repr_dims,
            self.args.kernel_size,
            groups,
            self.args.depth,
        ).to(self.device)
        print(f"动态初始化特征提取器, 变量分组: {groups} (move to {self.device})")

    def _freeze_pretrained_components(self, transferred_layers=None):
        """智能冻结策略（冻结解耦组件与迁移层）"""
        # 冻结趋势/季节性解耦组件
        always_freeze = [self.trd, self.srd]
        for component in always_freeze:
            for param in component.parameters():
                param.requires_grad = False

        # 对特征提取器进行智能冻结
        if self.feature_extractor is not None and transferred_layers:
            for layer_info in transferred_layers:
                if layer_info['type'] == 'group_conv':
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
                    component_name = layer_info['component']
                    if component_name == 'weight':
                        self.feature_extractor.final_conv.weight.requires_grad = False
                    elif component_name == 'bias':
                        self.feature_extractor.final_conv.bias.requires_grad = False

        print("冻结预训练组件（有监督预训练版）完成")

    def _freeze_transferred_layers(self, transferred_layers):
        """只冻结成功迁移的层"""
        for layer_info in transferred_layers:
            if layer_info['type'] == 'group_conv':
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
                for param in self.feature_extractor.final_conv.parameters():
                    param.requires_grad = False

    def _transfer_encoder_weights(self, pretrained_state_dict, tgt_encoder):
        """从state_dict迁移权重到目标编码器"""
        transferred_layers = []

        # 迁移分组卷积权重
        for layer_idx in range(len(tgt_encoder.group_convs)):
            tgt_group_conv = tgt_encoder.group_convs[layer_idx]

            # 只迁移深层（浅层重新初始化，保持可训练）
            if layer_idx >= getattr(self.args, 'freeze_start_layer', 2):
                for block_idx in range(len(tgt_group_conv)):
                    block = tgt_group_conv[block_idx]

                    conv1_key = f'feature_extractor.group_convs.{layer_idx}.{block_idx}.conv1.conv.weight'
                    conv2_key = f'feature_extractor.group_convs.{layer_idx}.{block_idx}.conv2.conv.weight'

                    if conv1_key in pretrained_state_dict and hasattr(block, 'conv1'):
                        src_weight = pretrained_state_dict[conv1_key]
                        if src_weight.shape == block.conv1.conv.weight.shape:
                            block.conv1.conv.weight.data = src_weight.clone()
                            transferred_layers.append({
                                'type': 'group_conv',
                                'layer_idx': layer_idx,
                                'block_idx': block_idx,
                                'component': 'conv1'
                            })

                    if conv2_key in pretrained_state_dict and hasattr(block, 'conv2'):
                        src_weight = pretrained_state_dict[conv2_key]
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

        if (final_conv_weight_key in pretrained_state_dict and
            pretrained_state_dict[final_conv_weight_key].shape == tgt_encoder.final_conv.weight.shape):
            tgt_encoder.final_conv.weight.data = pretrained_state_dict[final_conv_weight_key].clone()
            transferred_layers.append({'type': 'final_conv', 'component': 'weight'})

        if (final_conv_bias_key in pretrained_state_dict and
            pretrained_state_dict[final_conv_bias_key].shape == tgt_encoder.final_conv.bias.shape):
            tgt_encoder.final_conv.bias.data = pretrained_state_dict[final_conv_bias_key].clone()
            transferred_layers.append({'type': 'final_conv', 'component': 'bias'})

        return tgt_encoder, transferred_layers

    def _transfer_conv_block_weights(self, src_block, tgt_block):
        """迁移卷积块权重，返回是否成功迁移"""
        transferred = False

        if (hasattr(src_block, 'conv1') and hasattr(tgt_block, 'conv1') and
            src_block.conv1.conv.weight.shape == tgt_block.conv1.conv.weight.shape):
            tgt_block.conv1.conv.weight.data = src_block.conv1.conv.weight.data.clone()
            if src_block.conv1.conv.bias is not None:
                tgt_block.conv1.conv.bias.data = src_block.conv1.conv.bias.data.clone()
            transferred = True

        if (hasattr(src_block, 'conv2') and hasattr(tgt_block, 'conv2') and
            src_block.conv2.conv.weight.shape == tgt_block.conv2.conv.weight.shape):
            tgt_block.conv2.conv.weight.data = src_block.conv2.conv.weight.data.clone()
            if src_block.conv2.conv.bias is not None:
                tgt_block.conv2.conv.bias.data = src_block.conv2.conv.bias.data.clone()
            transferred = True

        return transferred

    def forward(self, x, pretrained_state_dict=None):
        """
        x: (batch_size, seq_len, input_size)
        始终返回预测结果（与自监督版不同，stage 1也输出完整预测）
        """
        batch_size, seq_len, input_size = x.shape

        # Stage 2: 权重迁移（仅执行一次）
        transferred_layers = []
        if pretrained_state_dict is not None and not getattr(self, '_is_transferred') and self.stage == 2:
            self.feature_extractor, transferred_layers = self._transfer_encoder_weights(
                pretrained_state_dict, self.feature_extractor
            )
            self._is_transferred = True

        # Stage 2: 智能冻结
        if self.stage == 2 and not hasattr(self, '_frozen'):
            self._freeze_pretrained_components(transferred_layers)
            self._frozen = True

        # 应用输入投影
        projected_vars = []
        for var_idx in range(input_size):
            var_data = x[:, :, var_idx:var_idx+1]  # [batch, seq_len, 1]
            projected_var = self.input_projection_list[var_idx](var_data)  # [batch, seq_len, hidden_dim]
            projected_vars.append(projected_var)
        x_projected = torch.cat(projected_vars, dim=-1)
        y = self.feature_extractor(x_projected)  # [batch, repr_dims, seq_len]

        # 提取趋势性成分特征
        trend = []
        for idx, mod in enumerate(self.trd.conv_layers):
            out = mod(y)
            if self.args.kernels[idx] != 1:
                out = out[..., :-(self.args.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))
        trend = torch.mean(torch.stack(trend), dim=0)
        trend = self.trend_norm(trend)

        # 提取季节性成分特征
        season = self.srd(y.transpose(1, 2))  # [B, T, repr_dims//2]
        season = self.season_norm(season)

        # CGU双向调制
        trend_modulated, season_modulated = self.cgu(trend, season)

        trend = self.repr_dropout(trend_modulated)
        season = self.repr_dropout(season_modulated)

        # 始终输出完整预测（不像自监督版在stage 1只返回表示）
        output = torch.cat([trend, season], dim=2)  # (batch_size, seq_len, repr_dims)
        output = self.feature_reducer(output.transpose(1, 2)).transpose(1, 2)  # (batch_size, seq_len, hidden_dims)
        output = self.projection(output)  # (batch_size, seq_len, 1)
        output = self.pool(output.transpose(1, 2)).transpose(1, 2)  # (batch_size, pred_len, 1)
        return output[:, -self.args.pred_len:, :]

    def compute_loss(self, batch_x, batch_y):
        """
        有监督预训练损失（MSE）
        batch_x: (batch_size, seq_len, input_size)
        batch_y: (batch_size, pred_len, 1)
        """
        outputs = self.forward(batch_x)
        loss = F.mse_loss(outputs, batch_y)
        return loss


class ELFNet_common_TS(nn.Module):
    """
    传统季节-趋势解耦方案（消融模型）
    使用与 DLinear 一致的滑动平均分解替代深度季节-趋势解耦模块，
    不进行多尺度趋势/季节提取及双向调制。
    """
    def __init__(self, args, device):
        super(ELFNet_common_TS, self).__init__()
        self.args = args
        self.device = device
        self.stage = 1  # 单阶段有监督训练

        # 动态初始化的组件
        self.input_projection_list = None
        self.feature_extractor = None

        # 滑动平均分解模块（与 DLinear 相同）
        self.decomposition = series_decomp(args.moving_avg)

        # 后续处理模块（与原 ELFNet 保持一致）
        self.feature_reducer = FeatureReducer(args, args.hidden_dims)
        self.projection = nn.Linear(args.hidden_dims, args.c_out, bias=True)
        self.pool = nn.AdaptiveAvgPool1d(output_size=args.pred_len)

    def _init_input_projections(self, num_vars, hidden_dim):
        """动态初始化输入投影层（与原 ELFNet 相同）"""
        self.input_projection_list = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_vars)
        ]).to(self.device)

    def _init_feature_extractor(self, groups):
        """动态初始化特征提取器（与原 ELFNet 相同）"""
        self.feature_extractor = MixedChannelConvEncoder(
            self.args.hidden_dims,
            self.args.repr_dims,
            self.args.kernel_size,
            groups,
            self.args.depth
        ).to(self.device)

    def forward(self, x, pretrained_state_dict=None):
        """
        输入 x 形状: (batch_size, seq_len, input_size)
        返回预测输出形状: (batch_size, pred_len, 1)
        """
        batch_size, seq_len, input_size = x.shape

        # 1. 输入投影（每个变量独立映射）
        projected = []
        for var_idx in range(input_size):
            var_data = x[:, :, var_idx:var_idx+1]               # (B, seq_len, 1)
            proj = self.input_projection_list[var_idx](var_data)  # (B, seq_len, hidden_dim)
            projected.append(proj)
        x_proj = torch.cat(projected, dim=-1)                   # (B, seq_len, input_size * hidden_dim)

        # 2. 特征提取
        y = self.feature_extractor(x_proj)                     # (B, repr_dims, seq_len)

        # 3. 传统季节-趋势分解（滑动平均）
        y_perm = y.transpose(1, 2)                              # (B, seq_len, repr_dims)
        seasonal, trend = self.decomposition(y_perm)            # 两者形状均为 (B, seq_len, repr_dims)

        # 4. 简单相加得到融合特征（无双向调制）
        fused = seasonal + trend                                 # (B, seq_len, repr_dims)

        # 5. 后续处理（与原 ELFNet 一致）
        fused = self.feature_reducer(fused.transpose(1, 2)).transpose(1, 2)  # (B, seq_len, input_size)
        fused = self.projection(fused)                          # (B, seq_len, 1)
        fused = self.pool(fused.transpose(1, 2)).transpose(1, 2)  # (B, pred_len, 1)

        return fused[:, -self.args.pred_len:, :]                # 取最后 pred_len 步


class SingleBandSeasonalDisentangler(nn.Module):
    """
    单频带可学习傅里叶层消融版本。
    使用单一 BandedFourierLayer（band=0, num_bands=1）替换原来的多频带融合结构，
    用于验证多频带设计在季节性解耦中的有效性。
    """
    def __init__(self, in_channels, out_channels, length=201):
        super(SingleBandSeasonalDisentangler, self).__init__()
        self.length = length
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 单频带可学习傅里叶层（覆盖全频带）
        self.fourier_layer = BandedFourierLayer(
            in_channels, out_channels, band=0, num_bands=1, length=length
        )

    def forward(self, x):
        """
        x: [B, T, D] 输入特征
        返回: [B, T, out_channels] 季节性表示
        """
        return self.fourier_layer(x)


class ELFNet_single_band_SRD(nn.Module):
    """
    单频带可学习傅里叶层版本消融模型（第三章消融实验）。
    相比有监督版ELFNet (ELFNet_supervised)，
    仅将MultiBandSeasonalDisentangler替换为SingleBandSeasonalDisentangler，
    其余结构完全相同。
    """
    def __init__(self, args, device):
        super(ELFNet_single_band_SRD, self).__init__()

        self.args = args
        self.device = device
        self.stage = 1  # 单阶段有监督训练

        # 动态初始化的输入投影层列表和特征提取器
        self.input_projection_list = None
        self.feature_extractor = None

        # Trend Representation Disentangler
        self.trd = TrendRepresentationDisentangler(args)

        # 单频带季节性解耦（与有监督版ELFNet的唯一区别）
        self.srd = SingleBandSeasonalDisentangler(
            in_channels=args.repr_dims,
            out_channels=args.repr_dims // 2,
            length=args.seq_len
        )

        self.repr_dropout = nn.Dropout(p=0.1)
        # 趋势/季节性表示特征归一化层
        self.trend_norm = nn.LayerNorm(args.repr_dims // 2)
        self.season_norm = nn.LayerNorm(args.repr_dims // 2)

        # 耦合门控单元
        self.cgu = CoupledGatingUnit(args.repr_dims // 2)

        # 特征还原与预测头
        self.feature_reducer = FeatureReducer(args, args.hidden_dims)
        self.projection = nn.Linear(args.hidden_dims, args.c_out, bias=True)
        self.pool = nn.AdaptiveAvgPool1d(output_size=args.pred_len)

    def _init_input_projections(self, num_vars, hidden_dim):
        """动态初始化输入投影层，为每个变量进行独立映射"""
        self.input_projection_list = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_vars)
        ]).to(self.device)
        print(f"动态初始化输入投影层: {num_vars} 个变量 -> 隐藏维度 {hidden_dim} (move to {self.device})")

    def _init_feature_extractor(self, groups):
        """动态初始化特征提取器"""
        self.feature_extractor = MixedChannelConvEncoder(
            self.args.hidden_dims,
            self.args.repr_dims,
            self.args.kernel_size,
            groups,
            self.args.depth
        ).to(self.device)
        print(f"动态初始化特征提取器, 变量分组: {groups} (move to {self.device})")

    def forward(self, x, pretrained_state_dict=None):
        """
        x: (batch_size, seq_len, input_size)
        返回: (batch_size, pred_len, 1)
        """
        batch_size, seq_len, input_size = x.shape

        # 应用输入投影
        projected_vars = []
        for var_idx in range(input_size):
            var_data = x[:, :, var_idx:var_idx+1]  # [batch, seq_len, 1]
            projected_var = self.input_projection_list[var_idx](var_data)  # [batch, seq_len, hidden_dim]
            projected_vars.append(projected_var)
        x_projected = torch.cat(projected_vars, dim=-1)
        y = self.feature_extractor(x_projected)  # [batch, repr_dims, seq_len]

        # 提取趋势性成分特征
        trend = []
        for idx, mod in enumerate(self.trd.conv_layers):
            out = mod(y)
            if self.args.kernels[idx] != 1:
                out = out[..., :-(self.args.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))
        trend = torch.mean(torch.stack(trend), dim=0)
        trend = self.trend_norm(trend)

        # 提取季节性成分特征（单频带可学习傅里叶层）
        season = self.srd(y.transpose(1, 2))  # [B, T, repr_dims//2]
        season = self.season_norm(season)

        # CGU双向调制
        trend_modulated, season_modulated = self.cgu(trend, season)

        trend = self.repr_dropout(trend_modulated)
        season = self.repr_dropout(season_modulated)

        # 还原与预测
        output = torch.cat([trend, season], dim=2)  # (batch_size, seq_len, repr_dims)
        output = self.feature_reducer(output.transpose(1, 2)).transpose(1, 2)  # (batch_size, seq_len, hidden_dims)
        output = self.projection(output)  # (batch_size, seq_len, 1)
        output = self.pool(output.transpose(1, 2)).transpose(1, 2)  # (batch_size, pred_len, 1)
        return output[:, -self.args.pred_len:, :]


class ELFNet_wo_CGU(nn.Module):
    """
    移除双向调制机制（CoupledGatingUnit）的消融模型（第三章消融实验）。
    相比自监督版ELFNet，仅移除CoupledGatingUnit，其余所有结构（TRD、SRD、head、
    augmentor、两阶段训练流程等）完全相同。
    """
    def __init__(self, args, device, stage=1):
        super(ELFNet_wo_CGU, self).__init__()

        self.args = args
        self.stage = stage
        self.device = device

        self.input_projection_list = None
        self.feature_extractor = None

        self.trd = TrendRepresentationDisentangler(args)

        # projection head
        self.head = nn.Sequential(
            nn.Linear(args.repr_dims // 2, args.repr_dims // 2),
            nn.ReLU(),
            nn.Linear(args.repr_dims // 2, args.repr_dims // 2)
        )

        self.srd = MultiBandSeasonalDisentangler(
            in_channels=args.repr_dims,
            out_channels=args.repr_dims // 2,
            num_bands=3,
            length=args.seq_len
        )

        self.repr_dropout = nn.Dropout(p=0.1)
        self.trend_norm = nn.LayerNorm(args.repr_dims // 2)
        self.season_norm = nn.LayerNorm(args.repr_dims // 2)

        # CGU已被移除——趋势与季节性表示之间不再进行双向调制

        self.feature_reducer = FeatureReducer(args, args.hidden_dims)
        self.projection = nn.Linear(args.hidden_dims, args.c_out, bias=True)
        self.pool = nn.AdaptiveAvgPool1d(output_size=args.pred_len)

        self.augmentor = DomainAugmentationFramework(args)
        self._is_transferred = False

    def _init_input_projections(self, num_vars, hidden_dim):
        self.input_projection_list = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_vars)
        ]).to(self.device)
        print(f"动态初始化输入投影层: {num_vars} 个变量 -> 隐藏维度 {hidden_dim}")

    def _init_feature_extractor(self, groups):
        self.feature_extractor = MixedChannelConvEncoder(
            self.args.hidden_dims, self.args.repr_dims,
            self.args.kernel_size, groups, self.args.depth,
        ).to(self.device)
        print(f"动态初始化特征提取器, 变量分组: {groups}")

    def _freeze_pretrained_components(self, transferred_layers=None):
        always_freeze = [self.trd, self.head, self.srd]
        for component in always_freeze:
            for param in component.parameters():
                param.requires_grad = False
        # 特征提取器冻结逻辑与原始ELFNet完全一致
        if self.feature_extractor is not None and transferred_layers:
            for layer_info in transferred_layers:
                if layer_info['type'] == 'group_conv':
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
                    component_name = layer_info['component']
                    if component_name == 'weight':
                        self.feature_extractor.final_conv.weight.requires_grad = False
                    elif component_name == 'bias':
                        self.feature_extractor.final_conv.bias.requires_grad = False
        print("冻结预训练组件（wo_CGU）完成")

    def _freeze_transferred_layers(self, transferred_layers):
        for layer_info in transferred_layers:
            if layer_info['type'] == 'group_conv':
                group_idx = layer_info['group_idx']
                layer_idx = layer_info['layer_idx']
                block_idx = layer_info['block_idx']
                if (group_idx < len(self.feature_extractor.group_convs) and
                    layer_idx < len(self.feature_extractor.group_convs[group_idx]) and
                    block_idx < len(self.feature_extractor.group_convs[group_idx][layer_idx])):
                    for param in self.feature_extractor.group_convs[group_idx][layer_idx][block_idx].parameters():
                        param.requires_grad = False
            elif layer_info['type'] == 'final_conv':
                for param in self.feature_extractor.final_conv.parameters():
                    param.requires_grad = False

    def _transfer_encoder_weights(self, pretrained_state_dict, tgt_encoder):
        transferred_layers = []
        for layer_idx in range(len(tgt_encoder.group_convs)):
            tgt_group_conv = tgt_encoder.group_convs[layer_idx]
            if layer_idx >= getattr(self.args, 'freeze_start_layer', 2):
                for block_idx in range(len(tgt_group_conv)):
                    block = tgt_group_conv[block_idx]
                    conv1_key = f'feature_extractor.group_convs.{layer_idx}.{block_idx}.conv1.conv.weight'
                    conv2_key = f'feature_extractor.group_convs.{layer_idx}.{block_idx}.conv2.conv.weight'
                    if conv1_key in pretrained_state_dict and hasattr(block, 'conv1'):
                        if pretrained_state_dict[conv1_key].shape == block.conv1.conv.weight.shape:
                            block.conv1.conv.weight.data = pretrained_state_dict[conv1_key].clone()
                            transferred_layers.append({'type': 'group_conv', 'layer_idx': layer_idx, 'block_idx': block_idx, 'component': 'conv1'})
                    if conv2_key in pretrained_state_dict and hasattr(block, 'conv2'):
                        if pretrained_state_dict[conv2_key].shape == block.conv2.conv.weight.shape:
                            block.conv2.conv.weight.data = pretrained_state_dict[conv2_key].clone()
                            transferred_layers.append({'type': 'group_conv', 'layer_idx': layer_idx, 'block_idx': block_idx, 'component': 'conv2'})

        fw_key = 'feature_extractor.final_conv.weight'
        fb_key = 'feature_extractor.final_conv.bias'
        if fw_key in pretrained_state_dict and pretrained_state_dict[fw_key].shape == tgt_encoder.final_conv.weight.shape:
            tgt_encoder.final_conv.weight.data = pretrained_state_dict[fw_key].clone()
            transferred_layers.append({'type': 'final_conv', 'component': 'weight'})
        if fb_key in pretrained_state_dict and pretrained_state_dict[fb_key].shape == tgt_encoder.final_conv.bias.shape:
            tgt_encoder.final_conv.bias.data = pretrained_state_dict[fb_key].clone()
            transferred_layers.append({'type': 'final_conv', 'component': 'bias'})
        return tgt_encoder, transferred_layers

    def _transfer_conv_block_weights(self, src_block, tgt_block):
        transferred = False
        if (hasattr(src_block, 'conv1') and hasattr(tgt_block, 'conv1') and
            src_block.conv1.conv.weight.shape == tgt_block.conv1.conv.weight.shape):
            tgt_block.conv1.conv.weight.data = src_block.conv1.conv.weight.data.clone()
            if src_block.conv1.conv.bias is not None:
                tgt_block.conv1.conv.bias.data = src_block.conv1.conv.bias.data.clone()
            transferred = True
        if (hasattr(src_block, 'conv2') and hasattr(tgt_block, 'conv2') and
            src_block.conv2.conv.weight.shape == tgt_block.conv2.conv.weight.shape):
            tgt_block.conv2.conv.weight.data = src_block.conv2.conv.weight.data.clone()
            if src_block.conv2.conv.bias is not None:
                tgt_block.conv2.conv.bias.data = src_block.conv2.conv.bias.data.clone()
            transferred = True
        return transferred

    def forward(self, x, pretrained_state_dict=None):
        batch_size, seq_len, input_size = x.shape

        transferred_layers = []
        if pretrained_state_dict is not None and not getattr(self, '_is_transferred') and self.stage == 2:
            self.feature_extractor, transferred_layers = self._transfer_encoder_weights(
                pretrained_state_dict, self.feature_extractor
            )
            self._is_transferred = True

        if self.stage == 2 and not hasattr(self, '_frozen'):
            self._freeze_pretrained_components(transferred_layers)
            self._frozen = True

        projected_vars = []
        for var_idx in range(input_size):
            var_data = x[:, :, var_idx:var_idx+1]
            projected_var = self.input_projection_list[var_idx](var_data)
            projected_vars.append(projected_var)
        x_projected = torch.cat(projected_vars, dim=-1)
        y = self.feature_extractor(x_projected)

        # 趋势性成分
        trend = []
        for idx, mod in enumerate(self.trd.conv_layers):
            out = mod(y)
            if self.args.kernels[idx] != 1:
                out = out[..., :-(self.args.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))
        trend = torch.mean(torch.stack(trend), dim=0)
        trend = self.trend_norm(trend)

        # 季节性成分
        season = self.srd(y.transpose(1, 2))
        season = self.season_norm(season)

        # CGU已移除：趋势与季节性表示直接使用，无双向调制
        trend = self.repr_dropout(trend)
        season = self.repr_dropout(season)

        if self.stage == 1:
            return trend, season
        else:
            output = torch.cat([trend, season], dim=2)
            output = self.feature_reducer(output.transpose(1, 2)).transpose(1, 2)
            output = self.projection(output)
            output = self.pool(output.transpose(1, 2)).transpose(1, 2)
            return output[:, -self.args.pred_len:, :]

    def compute_loss(self, batch_x, plot_dir, plot_augment_flag):
        if not self.args.plot_augment:
            plot_dir = None
        batch_x, positive_batch_x, negative_batch_x_list, trend_negative_batch_x_list, seasonal_negative_batch_x_list = self.augmentor.augment_batch(batch_x, plot_dir, plot_augment_flag)
        t_negative_batch_x_list = negative_batch_x_list + trend_negative_batch_x_list
        s_negative_batch_x_list = negative_batch_x_list + seasonal_negative_batch_x_list

        batch_x = batch_x.float().transpose(1, 2)
        output_t, output_s = self.forward(batch_x)

        positive_batch_x = positive_batch_x.transpose(1, 2).float()
        output_positive_t, output_positive_s = self.forward(positive_batch_x)

        output_negative_t_list = []
        for negative_batch_x in t_negative_batch_x_list:
            output_negative_t, _ = self.forward(negative_batch_x.transpose(1, 2).float())
            output_negative_t_list.append(output_negative_t)

        output_negative_s_list = []
        for negative_batch_x in s_negative_batch_x_list:
            _, output_negative_s = self.forward(negative_batch_x.transpose(1, 2).float())
            output_negative_s_list.append(output_negative_s)

        trend_loss = self._compute_trend_contrastive_loss(output_t, output_positive_t, output_negative_t_list)
        seasonal_loss = self._compute_seasonal_contrastive_loss(output_s, output_positive_s, output_negative_s_list)
        loss = trend_loss + self.args.alpha * seasonal_loss
        return loss

    def _compute_trend_contrastive_loss(self, anchor_t, pos_t, neg_t_list):
        B, seq_len, C = anchor_t.shape
        key_indices = [0, seq_len//4, seq_len//2, 3*seq_len//4, -1]
        if len(key_indices) > seq_len:
            key_indices = list(range(seq_len))
        losses = []
        for idx in key_indices:
            anchor_feat = F.normalize(self.head(anchor_t[:, idx, :]), dim=-1)
            pos_feat = F.normalize(self.head(pos_t[:, idx, :]), dim=-1)
            neg_feats = torch.stack([F.normalize(self.head(neg_t[:, idx, :]), dim=-1) for neg_t in neg_t_list], dim=1)
            losses.append(self.caculate_unified_contrastive_loss(anchor_feat, pos_feat, neg_feats))
        return torch.mean(torch.stack(losses))

    def _compute_seasonal_contrastive_loss(self, anchor_s, pos_s, neg_s_list):
        B, seq_len, C = anchor_s.shape
        anchor_freq = fft.rfft(anchor_s, dim=1)
        pos_freq = fft.rfft(pos_s, dim=1)
        anchor_amp, anchor_phase = self.convert_coeff(anchor_freq)
        pos_amp, pos_phase = self.convert_coeff(pos_freq)

        freq_bins = anchor_amp.shape[1]
        key_freq_indices = [0, freq_bins//4, freq_bins//2, -1]
        losses = []
        for freq_idx in key_freq_indices:
            if freq_idx >= freq_bins:
                continue

            neg_amp_feats = torch.stack([self.convert_coeff(fft.rfft(neg_s, dim=1))[0][:, freq_idx, :] for neg_s in neg_s_list], dim=1)
            amp_loss = self.caculate_unified_contrastive_loss(anchor_amp[:, freq_idx, :], pos_amp[:, freq_idx, :], neg_amp_feats)

            neg_phase_feats = torch.stack([self.convert_coeff(fft.rfft(neg_s, dim=1))[1][:, freq_idx, :] for neg_s in neg_s_list], dim=1)
            phase_loss = self.caculate_unified_contrastive_loss(anchor_phase[:, freq_idx, :], pos_phase[:, freq_idx, :], neg_phase_feats)

            losses.append(amp_loss + phase_loss)
        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0)

    def caculate_unified_contrastive_loss(self, anchor, pos, negs, temperature=None):
        if temperature is None:
            temperature = self.args.temperature
        anchor = F.normalize(anchor, dim=-1)
        pos = F.normalize(pos, dim=-1)
        negs = F.normalize(negs, dim=-1)
        logits = torch.cat([
            torch.sum(anchor * pos, dim=-1, keepdim=True),
            torch.sum(anchor.unsqueeze(1) * negs, dim=-1)
        ], dim=-1) / temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def convert_coeff(x, eps=1e-6):
        return torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2)), torch.atan2(x.imag, x.real + eps)
    
    