"""
    
    token2vec.py完成了词嵌入并写入到了token2vec.csv文件中

    这里只需要完成不同长度序列的padding工作

"""
import torch
from torch import nn
from util.tokenizer import model_token_padding
from util.token2vec import origin_model_token2vec
class tokenEmbedding(nn.Module):
    """
    just padding!
    """

    def __init__(self, d_model, max_len):
        
        super(tokenEmbedding, self).__init__()
        
        self.d_model = d_model # 备用
        self.max_len = max_len


    def forward(self, x):
        batch, length, d_m = x.size()
        shape = (batch, self.max_len, d_m)
        _x = torch.zeros(shape) # padding data would be masked, so don't worry about it
        _x[:, 0:length, :] = x
        _x[:, length:self.max_len, :] = torch.tensor(origin_model_token2vec[model_token_padding])
        x = _x       
        return x
