"""
    The fc1.weight of model_word2vec is the word embedding.
"""
import torch
from torch import nn
from sub_model2.model.my_word2vec import my_word2vec
from sub_model2.train import vocab_size, embedding_size

model = my_word2vec(vocab_size, embedding_size)
shape = (vocab_size, embedding_size)
model.load_state_dict(torch.load('saved/model_word2vec.pt'))

idx2vec = torch.tensor(shape)
for name, param in model.named_parameters():
    if name == 'fc1.weight':
        idx2vec = param.data.transpose(1, 0)
        break

model.load_state_dict(torch.load('saved/model_word2vec_tgt.pt'))
idx2vec_tgt = torch.tensor(shape)
for name, param in model.named_parameters():
    if name == 'fc1.weight':
        idx2vec_tgt = param.data.transpose(1, 0)
        break


class tokenEmbedding(nn.Module):
    """
        idx -> vec
    """

    def __init__(self, d_model, max_len, ty):
        
        super(tokenEmbedding, self).__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.ty = ty


    def forward(self, x):
        batch_size, length = x.size() # in fact, length is equal to max_len at here
        x_embedding = torch.zeros(batch_size*length, embedding_size)
        idxs = x.flatten()
        for i in range(0, len(idxs)):
            if self.ty == 'src_embedding':
                x_embedding[i:i+1, :] = idx2vec[idxs[i]]
            else:
                x_embedding[i:i+1, :] = idx2vec_tgt[idxs[i]]
        x_embedding = x_embedding.reshape(batch_size, length, embedding_size)
        # idx2vec: [vocab_size, d_model]
        return x_embedding
