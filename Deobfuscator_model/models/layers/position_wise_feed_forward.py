import torch
from torch import nn

class positionWiseFeedForward(nn.Module): # 编码器和解码器后面的全连接层
    def __init__(self, d_model, hidden, drop_prob=0.1): # hidden: ffn_hidden=2048 in conf.ipynb
        super(positionWiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, hidden) # todo: hidden = 2048 ? 可以变吗？
        self.linear_2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x