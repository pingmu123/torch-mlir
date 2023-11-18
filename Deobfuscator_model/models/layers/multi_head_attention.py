import numpy as np
import torch
from torch import nn
from models.layers.scale_dot_product_attention import scaleDotProductAttention 
from conf import *


class multiHeadAttention(nn.Module):
    
    def __init__(self, d_model, n_head):
        super(multiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = scaleDotProductAttention()
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        # 1.  q1 = Wq * a1, k1 = Wk * a1, k2 = Wk * a2, ...
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)  # 2 7 512 -> [[2, 7, 1], ..., [2, 7, 4]]
        
        # 3. dot product: 获取关联度和注意力分数
        # out, attention = self.attention(q, k, v, mask=None):
        #   TypeError: _forward_unimplemented() got an unexpected keyword argument 'mask'
        out, attention = self.attention(q, k, v) # out: [[2, 7, 1], ..., [2, 7, 4]],    attention: [2, 5, 7, 7]
        
        # 4. concat
        out = self.concat(out) # multi-head # [2, 7, 512]
        out = self.w_concat(out) # 学习关联度的处理？

        return out
        
        
    def split(self, tensor):
        """
        Split tensor by number of heads: tensor -> tensor list
            
            param tensor: [batch_size, length, d_model]
            
            n_head = 5: SN, OpId, topo, params, shape
            return: [[batch_size, length, d_tensor_SN], ..., [batch_size, length, d_tensor_shape]]
        """

        pos_1 = d_model_SN_size
        pos_2 = pos_1 + d_model_OpId_size
        pos_3 = pos_2 + d_model_topo_size
        pos_4 = pos_3 + d_model_params_size

        d_tensor_SN = tensor[:, :, 0:pos_1] 
        d_tensor_OpId = tensor[:, :, pos_1:pos_2]
        d_tensor_topo = tensor[:, :, pos_2: pos_3]
        d_tensor_params = tensor[:, :, pos_3: pos_4]
        d_tensor_shape = tensor[:, :, pos_4: d_model]
        
        return [d_tensor_SN, d_tensor_OpId, d_tensor_topo, d_tensor_params, d_tensor_shape]
    
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

            param tensor: [[batch_size, length, d_tensor_SN], ..., [batch_size, length, d_tensor_shape]]
            
            return: [batch_size, length, d_model]
        """
        d_tensor_SN, d_tensor_OpId, d_tensor_topo, d_tensor_params, d_tensor_shape = tensor

        tensor = torch.concatenate((d_tensor_SN, d_tensor_OpId, d_tensor_topo, d_tensor_params, d_tensor_shape), axis=2)
        
        return tensor