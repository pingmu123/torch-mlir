from torch import nn
# import sys

from models.embedding.positional_encoding import positionalEncoding
from models.embedding.token_embedding import tokenEmbedding

class transformerEmbedding(nn.Module):
    """
    transformer_embedding = pos_emdding + token_embedding
    """

    def __init__(self, d_model, max_len, drop_prob, device, ty):
        """
        class for word embedding

        :parms d_model: dimensions of model
        """

        super(transformerEmbedding, self).__init__()
        self.tok_emb = tokenEmbedding(d_model, max_len, ty)
        self.pos_emb = positionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
    