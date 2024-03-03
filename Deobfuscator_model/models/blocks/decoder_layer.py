from torch import nn

from models.layers.layer_norm import layerNorm
from models.layers.multi_head_attention import multiHeadAttention
from models.layers.position_wise_feed_forward import positionWiseFeedForward


class decoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(decoderLayer, self).__init__()
        self.self_attention = multiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = layerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # enc-dec attention
        self.enc_dec_attention = multiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = layerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = positionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = layerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, tgt_mask, src_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=tgt_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(_x + x)

        if enc is not None:
            # 3. compute encoder-decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask) # Note params
           
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(_x + x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(_x + x)

        return x