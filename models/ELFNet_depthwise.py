import torch 
from torch import nn
import torch.nn.functional as F
from ELFNet.ELFNet import TrendFeatureDisentangler, BandedFourierLayer,FeatureReducer
import torch.fft as fft
from layers.depthwise import DepthwiseNet
from utils.augmentation import augment
import numpy as np

import warnings

warnings.filterwarnings('ignore')
    

class ELFNet_depthwise(nn.Module):
    def __init__(self,args, target, input_size, device, groups=None,stage2=False):
        ### 调用LMLFNet传入的input_size就是n_c(变量数)；depth是特征提取器的深度
        super(ELFNet_depthwise, self).__init__()

        self.args = args
        self.target=target # 目标变量
        self.stage2 = stage2
        self.device = device
        
        
        self.dwn = DepthwiseNet(self.target, input_size, args.depth, kernel_size=args.kernel_size, dilation_c=args.dilation_c)
        self.pointwise = nn.Conv1d(input_size,args.repr_dims, 1)
        
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
            self.dwn = self.dwn.cuda()
            self.pointwise = self.pointwise.cuda()
            self.tfd = self.tfd.cuda()
            self.head = self.head.cuda()
            self.sfd = self.sfd.cuda() 
            self.feature_reducer = self.feature_reducer.cuda()
            self.projection = self.projection.cuda()
            self.pool = self.pool.cuda()

            # 微调阶段冻结 主干特征提取器(包括attention layer、input_fc、feature_extractor )和解耦器(包括tfd和sfd)的参数
            if stage2:
                for param in self.dwn.parameters():
                    param.requires_grad = False
                for param in self.pointwise.parameters():
                    param.requires_grad = False
                for param in self.tfd.parameters():
                    param.requires_grad = False
                for param in self.head.parameters():
                    param.requires_grad = False
                for param in self.sfd.parameters():
                    param.requires_grad = False

    
    '''
    前向传播函数。
    '''
    def forward(self, x):# 输入的x的形状为 b,input_size,seq_len
        
        batch_size, input_size , seq_len = x.shape
        
        ## x: (batch_size, T,input_size),reshape调整形状以适应 nn.Linear
        y = self.dwn(x)
        y = self.pointwise(y)
        
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