import torch as th
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.depthwise import DepthwiseNet


class ADDSTCN(nn.Module):
    def __init__(self, configs):
        super(ADDSTCN, self).__init__()

        self.target=configs.target
        input_size = configs.enc_in
        num_levels = configs.depth
        kernel_size = configs.kernel_size
        dilation_c = configs.dilation_c
        self.dwn = DepthwiseNet(self.target, input_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = nn.Conv1d(input_size, 1, 1)

        self._attention = th.ones(input_size,1)
        self._attention = Variable(self._attention, requires_grad=False)

        self.fs_attention = th.nn.Parameter(self._attention.data)
        
        if configs.use_gpu:
            self.dwn = self.dwn.cuda()
            self.pointwise = self.pointwise.cuda()
            self._attention = self._attention.cuda()
                  
                  
    def init_weights(self):
        self.pointwise.weight.data.normal_(0, 0.1)       
        
    def forward(self, x_enc ,x_mark_enc, x_dec, x_mark_dec):
        y1=self.dwn(x_enc*F.softmax(self.fs_attention, dim=0))
        y1 = self.pointwise(y1) 
        return y1.transpose(1,2)