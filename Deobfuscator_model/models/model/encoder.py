from torch import nn

from models.blocks.encoder_layer import encoderLayer
from models.embedding.transformer_embedding import transformerEmbedding


class Encoder(nn.Module):

    # todo: transformerEmbedding: vocab_size and enc_voc_size
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = transformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([encoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x