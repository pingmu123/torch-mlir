"""
1. 随机生成足够多的DNN模型

2. 将上述DNN模型转化为torch.mlir表示

3. 由上述torch.mlir表示得到混淆后的torch.mlir表示

4. 将3和2分别作为样本的输入和label得到数据集并存入.csv文件中

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


class RandomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d() # random
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), (2, 2), stride=(2, 2))
        return x