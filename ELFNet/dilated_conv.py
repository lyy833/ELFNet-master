import torch
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

class MixedChannelConvEncoder(nn.Module):
    def __init__(self, hidden_dims, repr_dims, kernel_size, groups, depth):
        super().__init__()

        self.groups = groups
        
        # Define conv layers for all groups using dilated convolution
        self.group_convs = nn.ModuleList([
            nn.Sequential(*[
                ConvBlock(
                    hidden_dims if i > 0 else len(group),
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
        batch_size, seq_len, input_size = x.size()
        
        conv_results = []

        # Process each group using dilated convolution
        for idx, group in enumerate(self.groups):
            group_features = x[:, :, group].permute(0, 2, 1)  # Shape: (batch_size, num_features, seq_len)
            conv_result = self.group_convs[idx](group_features)
            conv_results.append(conv_result)
        
        # Concatenate all convolution results along the channel dimension
        conv_results = torch.cat(conv_results, dim=1)  # Shape: (batch_size, total_channels, seq_len)
        
        # Apply final convolution
        output = self.final_conv(conv_results)
        
        return output # Shape: (batch_size, channels, seq_len)


class CI_DilatedConvEncoder(nn.Module):
    """通道独立的空洞卷积编码器"""
    def __init__(self, in_channels, hidden_dims, repr_dims, depth, kernel_size):
        super().__init__()
        
        # 使用共享的DilatedConvEncoder处理所有变量
        self.shared_encoder = DilatedConvEncoder(
            in_channels, hidden_dims, repr_dims, depth, kernel_size
        )
        
    def forward(self, x):
        """
        x: [batch_size, hidden_dim, seq_len] - 单个变量的投影后表示
        返回: [batch_size, repr_dims, seq_len]
        """
        return self.shared_encoder(x)

