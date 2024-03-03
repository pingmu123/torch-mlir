import torch
from model.my_word2vec import my_word2vec
from train import vocab_size, embedding_size


model = my_word2vec(vocab_size, embedding_size)

model.load_state_dict(torch.load('saved/model_word2vec.pt'))

shape = (vocab_size, embedding_size)
idx2vec =torch.tensor(shape)
for name, param in model.named_parameters():
    if name == 'fc1.weight':
        idx2vec = param.data.transpose(1, 0)
        break