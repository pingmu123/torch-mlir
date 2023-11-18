import torch
import math
from torch import nn
from conf import *

class scaleDotProductAttention(nn.Module): # self-attention
    """
    Compute scale dot product attention
    
        query: given sectence that we focused on(decoder)
    
        key: every sentence to check relationship with Query(encoder) 
        
        value: every sentence same with key(encoder)
        
        q1 = Wq * a1, k1 = Wk * a1, k2 = Wk * a2, ...
        
        score a1,a1: q1 * k1 * v1
        score a1,a2: q1 * k2 * v2
        ...
        
    """
    def __init__(self):
        super(scaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, mask=None, eps=1e-12):
        # input: q, k v
        # their shape: [[batch_size, length, d_tensor_SN], ..., [batch_size, length, d_tensor_shape]]

        # 1. q * k'
        score = [] 
        for i in range(0, len(q)):
            score.append((q[i] @ k[i].transpose(1, 2)) / math.sqrt(d_model)) # @: 矩阵乘
        
        # 2. apply masking(opt)
        # tensor_list to tensor and transpose
        score = torch.stack(score).transpose(0,1) # [] -> [5, 2, 7, 7] -> [2, 5, 7, 7]
        if mask is not None:
            score = score.masked_fill(mask==0, -inf) # 将0的地方替换为极小数，表示不用注意它们
        
        # 3.  softmax or others
        score = self.softmax(score)
        
        # 4. value
        # transpose and tensor to tensor_list
        score = score.transpose(0,1) # [2, 5, 7, 7] -> [5, 2, 7, 7]
        shape = score.shape
        score_list = []
        for i in range(0, shape[0]):
            score_list.append(score[i])
        score = score.transpose(0,1) # 恢复
        values = []
        for i in range(0, len(v)):
            values.append(score_list[i] @ v[i])
        
        score = score.transpose(0,1) # [5, 2, 7, 7] -> [2, 5, 7, 7]
        return values, score  # score: 关联度 v: 注意力分数
    
        # 暂时这样理解吧 2023.10.11