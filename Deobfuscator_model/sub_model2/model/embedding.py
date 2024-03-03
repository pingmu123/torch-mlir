# import torch
# from torch import nn
# from util.tokenizer import model_token_padding

# class tokenEmbedding(nn.Module):
#     """
#         just padding
#     """

#     def __init__(self, max_len):

#         super().__init__()

#         self.max_len = max_len

#     def forward(self, x):

#         batch, length, d_embedding = x.size()
#         if length < self.max_len:
            

    