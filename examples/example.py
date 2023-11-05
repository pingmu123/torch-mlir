import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


class example(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(6 * 3 * 3, 2)  # 5*5 from image dimension
        self.fc2 = nn.Linear(2, 3)
        self.train(False)

    def forward(self, x):
        # input shape is 1x1x28x28
        # Max pooling over a (2, 2) window, if use default stride, error will happen
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=(2, 2))
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


net = example()
print(net)
# compile to torch mlir
# NCHW layout in pytorch
print("================")
print("origin torch mlir")
print("================")
module = torch_mlir.compile(net, torch.ones(1, 1, 10, 10), output_type="torch")
print(module.operation.get_asm(large_elements_limit=10))
file = open("example.mlir", "w")
file.write(module.operation.get_asm())
file.close()
