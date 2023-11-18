import torch
from torch import nn
import numpy as np

import sys
sys.path.append('/home/pingmu123/torch-mlir/Deobfuscator_model')
from conf import *

from models.model.encoder import Encoder
from models.model.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, tgt_voc_size, d_model, n_head, max_len_src, max_len_tgt, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        
        # self.tgt_bos_idx = tgt_bos_idx # tgt 开始标记: origin_model_token_padding

        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len_src,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               tgt_voc_size=tgt_voc_size,
                               max_len=max_len_tgt,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        
    def make_src_mask(self, src):
        # difference: tokenEmbedding had finished, so src's shape: [batch, seq_len, d_model]

        # src_mask = (src != self.src_pad_idx) # get src_mask tensor: [True, ..., False, ..., False]
        shape = src.shape
        mask_shape = (shape[0], max_len_src)
        src_mask = np.ones(mask_shape)
        src_mask[:, shape[1]:max_len_src] = 0
        src_mask = torch.tensor(src_mask)
        src_mask = (src_mask != 0)
       
        src_mask = src_mask.unsqueeze(1).unsqueeze(2) # src: [2, 7]  score:[2, 5, 7, 7]  -> src_mask: [2, 1, 1, 7]
        # if mask is not None:
        #   score = score.masked_fill(mask == 0, -10000)  
        #   mask 7 and score 7*7： mask=[1, 1, 1, 1, 1, 0, 0]  score后2列全为-10000
        return src_mask
    
    def make_tgt_mask(self, tgt):
        # difference of src and tgt
        
        # tgt_pad_mask = (tgt != self.tgt_pad_idx)
        shape = tgt.shape
        mask_shape = (shape[0], max_len_tgt)
        tgt_pad_mask = np.ones(mask_shape)
        tgt_pad_mask[:, shape[1]:max_len_tgt] = 0
        tgt_pad_mask = torch.tensor(tgt_pad_mask)
        tgt_pad_mask = (tgt_pad_mask != 0)
        tgt_pad_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(3)
        
        # tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones(max_len_tgt, max_len_tgt)).type(torch.ByteTensor).to(self.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
        
    def forward(self, src, tgt): 
        # src and tgt: [batch, max_len](max_len = seq_len + pad_len) 
        # ex: 
        #   wonna take you baby take me higher pad pad pad
        #   gonna tiga take me take higher pad pad pad
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        if(src.shape[1] > max_len_src):
            print("length of input > max_len, exit!")
            exit()
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        return output
    

   
    # 1. shape of src and tgt: [batch, seq_len]
    # 2. why unsqueeze？for score mask
