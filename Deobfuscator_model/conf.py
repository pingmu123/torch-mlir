# model parameters setting
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 1
max_len_src = 64
max_len_tgt = max_len_src * 10
# max_len = 100
d_model = 512

n_head = 5
d_model_SN_size = 1
d_model_OpId_size = 64
d_model_topo_size = 16
d_model_shape_size = 4
d_model_other_size = d_model_SN_size + d_model_OpId_size + d_model_topo_size + d_model_shape_size
d_model_params_size = d_model - d_model_other_size

n_layers = 6
ffn_hidden = 2048
drop_prob = 0.1


init_lr = 0.1
factor = 0.9 # 学习率因子
adam_eps = 5e-9 # 数值稳定性，避免除0
patience = 10 # loss不再减小或增大的累计次数
# warmup = 100 # 每100次进行lr的更新？
# epoch = 1000
warmup = 10 # 每100次进行lr的更新？
epoch = 50
clip = 1.0 # 梯度裁剪参数
weight_decay = 5e-4 # 权重衰减（正则化）
# inf = float('inf') # 无穷大
inf = 10000



ob_model_vocab_size = 64 # todo
origin_model_vocab_size = d_model

