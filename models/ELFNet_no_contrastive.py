import torch 
from torch import nn
from ELFNet.ELFNet import FeatureReducer,TrendFeatureDisentangler,BandedFourierLayer
from ELFNet.dilated_conv import MixedChannelConvEncoder

import warnings

warnings.filterwarnings('ignore')

class ELFNet_no_contrastive(nn.Module):
    def __init__(self,args, target, input_size, device, groups=None,stage2=False):
        ### 调用CLMLFNet传入的input_size就是n_c(变量数)；depth是特征提取器的深度
        super(ELFNet_no_contrastive, self).__init__()

        self.args = args
        self.target=target # 目标变量
        self.device = device

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

    
    '''
    前向传播函数。
    '''
    def forward(self, x):# 输入的x的形状为 b,input_size,seq_len
        
        if  self.args.compare is not None:
            y = self.feature_extractor(x) 
        
        else:
            y = self.feature_extractor(x.transpose(1,2))
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

        output = torch.cat([trend, season], dim=2) # (batch_size,seq_len,repr_dims)
        output = (self.feature_reducer(output.transpose(1, 2))).transpose(1,2) # b  t input_size 

        output = self.projection(output)  # 应用全连接层，使维度从 (batch_size, t, input_size) 转换为 (batch_size, t, 1)
        output = self.pool(output.transpose(1, 2)).transpose(1, 2)  # 使用自适应平均池化调整时间步数
        return output[:, -self.args.pred_len:, :]    
