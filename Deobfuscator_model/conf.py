# model parameters setting
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

batch_size = 4
max_len = 512
max_len_src = 512
max_len_tgt = max_len_src // 4
# max_len = 100
d_model = 512

n_head = 4
# n_head = 5
# d_model_SN_size = 1
# d_model_OpId_size = 64
# d_model_topo_size = 16
# d_model_shape_size = 4
# d_model_other_size = d_model_SN_size + d_model_OpId_size + d_model_topo_size + d_model_shape_size
# d_model_params_size = d_model - d_model_other_size

n_layers = 8
ffn_hidden = 1024
drop_prob = 0.1


init_lr = 0.001
factor = 0.9 # 学习率因子
adam_eps = 5e-9 # 数值稳定性，避免除0
patience = 10 # loss不再减小或增大的累计次数
# warmup = 100 # 每100次进行lr的更新？
# epoch = 1000
warmup = 10
epoch = 100
clip = 1.0 # 梯度裁剪参数
weight_decay = 5e-4 # 权重衰减（正则化）
# inf = float('inf') # 无穷大
inf = 100000.0

origin_model_vocab_size = d_model

padding_shape = (batch_size, 3, 224, 224)


n_gram = 3 # the count of topo params is 8, other position less than it

