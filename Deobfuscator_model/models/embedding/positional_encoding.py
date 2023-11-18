import torch
from torch import nn

class positionalEncoding(nn.Module):
    """
    compute sinusoid encoding.

    # PE(pos, 2i) = sin(pos/(10000)**(2i/d_model))
    # PE(pos, 2i + 1) = cos(pos/((10000)**(2i/d_model)))

    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(positionalEncoding, self).__init__()

        # same size with input matrix
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # no computing gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1) # 1d -> 2D: word's position?

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model
        # "step=2" means 2*i

        self.encoding[:, 0::2] = torch.sin(pos/(10000**(_2i / d_model))) # even pos
        self.encoding[:, 1::2] = torch.cos(pos/(10000**(_2i / d_model))) # odd pos
        self.max_len = max_len

    def forward(self, x):
        # self.encoding
        # [max_len=xxx, d_model=512] # max_len = max number of Ops
        
        # batch_size, seq_len, d_m = x.size()
        # [batch_size=128, seq_len=30]

        return self.encoding[:self.max_len, :] # align: use max_len
        # [seq_len=30, d_model=512]
        
        # it will add with tok_emb: [128, 30, 512]

        # in a batch, the seq_len of inputs are equal(padding when not equal) 


