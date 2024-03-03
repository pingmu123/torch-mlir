import torch
from torch import nn

class my_word2vec(nn.Module):
    """
        model of word2vec, here we use skip-grim.
    """

    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # input x: [batch, 1, vocab_size]
        x = x.type(torch.float32) # int -> float
        x = self.fc1(x) # x: [batch, 1, embedding_size]
        x = self.fc2(x) # x: [batch, 1, vocab_size]

        # x = self.softmax(x) # nn.CrossEntropyLoss has softmax operation

        return x