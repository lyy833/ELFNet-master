from torch import nn
import torch.nn.functional as F


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
    def __init__(self, in_channels, hidden_dims,repr_dims, kernel_size, dilation, first=False,final=False,mixed=True):
        super().__init__()
        self.first = first
        self.final = final
        self.mixed = mixed
        self.conv1 = SamePadConv(in_channels, hidden_dims, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(hidden_dims, hidden_dims, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(hidden_dims, repr_dims, 1) 
    
    def forward(self, x):
        residual = x 
        if self.first is True:
            x = self.conv1(x)
            x = F.gelu(x)
            return x ###第一层不残差连接
        else:
            if self.final is True: 
                x = self.conv2(x)
                x = F.gelu(x)
                x = x + residual
                if self.mixed == False:
                    x = self.projector(x)
                return x 
            else:
                x = self.conv2(x)
                x = F.gelu(x)
                return x + residual
        

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels,hidden_dims,repr_dims,depth, kernel_size):
        '''
        这里定义的空洞/扩张卷积编码器DilatedConvEncoder在模型中充当特征提取器。
        它使用空洞卷积层来捕获输入数据中的多尺度上下文。根据需要，它可以从特定的层中提取中间输出，或者只提供最后一层的输出。
        '''
        super().__init__()

        self.net = nn.Sequential(*[
            ConvBlock(
                hidden_dims if i > 0 else in_channels,
                hidden_dims,
                repr_dims,
                kernel_size=kernel_size,
                dilation=2**i,
                first=(i==0),
                final=(i==depth-1),
                mixed=False
            )
            for i in range(depth) ### 从0到depth-1的层，依次进行卷积操作，并使用gelu激活函数
        ])
        
    def forward(self, x):
        return self.net(x)
