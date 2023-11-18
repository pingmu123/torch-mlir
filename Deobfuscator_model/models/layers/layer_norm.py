import torch
from torch import nn
class layerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12): # avoid to divide 0
        super(layerNorm, self).__init__() # 自己创建的类，然后确保初始化工作进行
        self.gamma = nn.Parameter(torch.ones(d_model)) # 每一个特征维度的放缩量均初始化为1，平移量为0
        self.beta = nn.Parameter(torch.zeros(d_model)) # 特征维度大小为d_model
        # nn.Parameter: 可学习的参数
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 计算每行的平均值并保持信息（-1指在最后一个维度上进行操作）
        var = x.var(-1, unbiased=False, keepdim=True) # 计算方差且不适用无偏估计（训练时）
        
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out