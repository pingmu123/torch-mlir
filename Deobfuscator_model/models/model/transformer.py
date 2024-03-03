import torch
from torch import nn
import numpy as np

import sys
sys.path.append('/home/pingmu123/torch-mlir/Deobfuscator_model')
from conf import *

from models.model.encoder import Encoder
from models.model.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, padding_idx, beginning_idx, ending_idx, tgt_voc_size, d_model, n_head, max_len_src, max_len_tgt, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        
        self.padding_idx = padding_idx
        self.beginnging_idx = beginning_idx
        self.ending_idx = ending_idx
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt

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
        
        # get src_mask tensor: [True, ..., False, ..., False] (while batch_size == 1)
        src_mask = (src != self.padding_idx) 
        src_mask = src_mask.unsqueeze(1).unsqueeze(2) # for score computing
    
        return src_mask
    
    def make_tgt_mask(self, tgt):
        # difference of src and tgt
        
        tgt_pad_mask = (tgt != self.padding_idx)
        tgt_pad_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(3)
        
        tgt_sub_mask = torch.tril(torch.ones(self.max_len_tgt, self.max_len_tgt)).type(torch.ByteTensor).to(self.device)
        
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        return tgt_mask
        
    def forward(self, src, tgt): 
        # src and tgt: [batch, max_len](max_len = seq_len + pad_len) 
        # ex: 
        #   wonna take you baby take me higher pad pad pad
        #   gonna tiga take me take me higher pad pad pad

        if(src.shape[1] > self.max_len_src):
            print('src_length:', src.shape[1])
            print("length of input > max_len_src, exit!")
            exit()
        
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_src = self.encoder(src, src_mask)

        # Are train and inference different?
        if self.training:
            output = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        else:
            print('inference begin')
            shape = (batch_size, self.max_len_tgt)
            tgt_eval = torch.zeros(shape, dtype=torch.int64)
            tgt_eval[:, 0:1] = self.beginnging_idx
            tgt_eval[:, 1:] = self.padding_idx
            tgt_eval_mask = torch.zeros(shape)
            tgt_eval_mask[:, 0:1] = 1
            tgt_eval_mask = (tgt_eval_mask != 0)
            tgt_eval_mask = tgt_eval_mask.unsqueeze(1).unsqueeze(2)
            for j in range(1, self.max_len_tgt):
                output = self.decoder(tgt_eval, enc_src, tgt_eval_mask, src_mask)
                for k in range(0, batch_size):
                    temp_tgt_eval = output[k].max(dim=-1)[1] # temp_tgt_eval: [length, ]
                    tgt_eval[k][j] = temp_tgt_eval[j]
                    
                tgt_eval_mask[:, :, :, j:j+1] = True
                # todo: 遇到'<EOS>'时可停止
            print('inference end')
        return output
    

   
    # 1. shape of src and tgt: [batch, seq_len]
    # 2. why unsqueeze？for score mask
