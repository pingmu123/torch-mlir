from torch import nn

from models.blocks.decoder_layer import decoderLayer
from models.embedding.transformer_embedding import transformerEmbedding


class Decoder(nn.Module):
    def __init__(self, tgt_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = transformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        device=device)

        self.layers = nn.ModuleList([decoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, tgt_voc_size)

    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        tgt = self.emb(tgt)

        for layer in self.layers:
            tgt = layer(tgt, enc_src, tgt_mask, src_mask)

        # pass to LM head
        output = self.linear(tgt) 
        return output