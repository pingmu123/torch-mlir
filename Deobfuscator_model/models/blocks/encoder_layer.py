from torch import nn

from models.layers.layer_norm import layerNorm
from models.layers.multi_head_attention import multiHeadAttention
from models.layers.position_wise_feed_forward import positionWiseFeedForward


class encoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(encoderLayer, self).__init__()
        self.attention = multiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = layerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = positionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = layerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask): # src_mask:
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(_x + x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout1(x)
        x = self.norm1(_x + x)

        return x