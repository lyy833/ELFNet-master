import torch 
from torch import nn
import torch.nn.functional as F
from ELFNet.ELFNet import FeatureReducer
from ELFNet.dilated_conv import MixedChannelConvEncoder
from utils.augmentation import augment
import numpy as np
import matplotlib.pyplot as plt
import os

import warnings

warnings.filterwarnings('ignore')

class LMLFNet_no_disentanglement(nn.Module):
    def __init__(self,args, target, input_size, device, groups=None,stage2=False):
        ### 调用CLMLFNet传入的input_size就是n_c(变量数)；depth是特征提取器的深度
        super(LMLFNet_no_disentanglement, self).__init__()

        self.args = args
        self.target=target # 目标变量
        self.stage2 = stage2
        self.device = device
        
    
        
        
        
        self.feature_extractor = MixedChannelConvEncoder(
                args.hidden_dims,
                args.repr_dims,
                args.kernel_size,
                groups,
                args.depth
            )
        
        self.head = nn.Sequential(
            nn.Linear(args.repr_dims, args.repr_dims),
            nn.ReLU(),
            nn.Linear(args.repr_dims , args.repr_dims)
        )
        self.feature_reducer= FeatureReducer(args,input_size)#将表示的维度从repr_dims  映射回到 input_size
        
        self.projection = nn.Linear(input_size, args.c_out, bias=True)  # 新增的全连接层
        self.pool = nn.AdaptiveAvgPool1d(output_size=args.pred_len )  # 自适应平均池化层

        if device == 'cuda:{}'.format(self.args.gpu):
            self.feature_extractor = self.feature_extractor.cuda()
            self.head = self.head.cuda()
            self.feature_reducer = self.feature_reducer.cuda()
            self.projection = self.projection.cuda()
            self.pool = self.pool.cuda()

            # 微调阶段冻结 主干特征提取器(包括、input_fc、feature_extractor )和解耦器(包括tfd和sfd)的参数
            if stage2:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                for param in self.head.parameters():
                    param.requires_grad = False

    
    '''
    前向传播函数。
    '''
    def forward(self, x):# 输入的x的形状为 b,input_size,seq_len
        
        batch_size, input_size , seq_len = x.shape
        
        ## x: (batch_size, T,input_size),reshape调整形状以适应 nn.Linear
        if  self.args.compare is not None:
            y = self.feature_extractor(x) 
        else:
            y = self.feature_extractor(x.transpose(1,2))
       
        output = y.transpose(1, 2)


        if not self.stage2:
            return output
        else:
            output = (self.feature_reducer(output.transpose(1, 2))).transpose(1,2) # b  t input_size 

            output = self.projection(output)  # 应用全连接层，使维度从 (batch_size, t, input_size) 转换为 (batch_size, t, 1)
            output = self.pool(output.transpose(1, 2)).transpose(1, 2)  # 使用自适应平均池化调整时间步数
            return output[:, -self.args.pred_len:, :]    
    
    
    def caculate_type1_loss(self, anchor, pos, negs):
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
    
    def caculate_type2_loss(self, z1, z2):
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
        batch_x, positive_batch_x, negative_batch_x_list= augment(batch_x,batch_y,self.args.num_augment,folder_path,plot=False)
        
        rand_idx = np.random.randint(0, batch_x.shape[1]) 
        
        # 原样本
        batch_x = batch_x.to(torch.float32)
        batch_x = batch_x.transpose(1, 2).to(self.device)
        output= self.forward(batch_x) # (b,seq_len,repr_dims) 
        output_type1 = F.normalize(self.head(output[:, rand_idx]), dim=-1)
        
        # 正样本
        positive_batch_x = torch.from_numpy(positive_batch_x.astype('float32'))
        positive_batch_x = positive_batch_x.transpose(1, 2)
        output_positive= self.forward(positive_batch_x.float().to(self.device)) # (b,seq_len,repr_dims/2) 
        output_positive_type1 = F.normalize(self.head(output_positive[:, rand_idx]), dim=-1) #(batch_size,seq_len,repr_dims/2)
        
        # 所有负样本
        output_negative_type1_list = []
        for negative_batch_x in negative_batch_x_list:
            # 计算每个负样本的趋势性输出
            negative_batch_x = negative_batch_x.to(torch.float32)
            negative_batch_x = negative_batch_x.transpose(1, 2)
            output_negative= self.forward(negative_batch_x.float().to(self.device)) # (b,seq_len,repr_dims/2) 
            output_negative_type1 = F.normalize(self.head(output_negative[:, rand_idx]), dim=-1) #(batch_size,repr_dims/2)
            output_negative_type1_list.append(output_negative_type1)
        output_negative_type1_all = torch.stack(output_negative_type1_list,dim=0) # (k,b,repr_dims/2)   其中k是负样本个数，k = num_augment
        output_negative_type1_all = output_negative_type1_all.transpose(0,1) # (b,k,repr_dims/2)
        #使用离散傅里叶变换等处理得到的原样本和正样本季节性输出矩阵
        output_type2 = F.normalize(output, dim=-1) # (b,seq_len,repr_dims/2)
        output_positive_type2 = F.normalize(output_positive, dim=-1) 
              
        # 计算两种类型对比损失
        type1_loss = self.caculate_type1_loss(output_type1,output_positive_type1,output_negative_type1_all)
        type2_loss = self.caculate_type2_loss(output_type2, output_positive_type2)
        
        # 计算总对比损失
        loss =   type1_loss + self.args.alpha * type2_loss
        
        return loss