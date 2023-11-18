"""
生成数据集：

    1. 随机生成足够多的DNN模型

    2. 将上述DNN模型转化为torch.mlir表示

    3. 由上述torch.mlir表示得到混淆后的torch.mlir表示

    4. 将3和2分别作为样本的data和label得到数据集并存入data.csv文件中

"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
csv_file1 = csv_file_path + "data.csv"


class RandomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,6,5) # random
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv(x)), (2, 2), stride=(2, 2))
        return x

net = RandomNet()
net.eval()
print(net)

# compile to torch mlir
origin_model_module = torch_mlir.compile(net, torch.ones(1, 1, 28, 28), output_type="torch")

ob_model_module = origin_model_module
torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    ob_model_module,
    "builtin.module(func.func(torch-branch-layer))",
    "BranchLayer",
)
with open(csv_file1, 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    ob_model = ob_model_module.operation.get_asm().replace(","," ")
    ob_model = ob_model.replace("\n"," EOVector")
    origin_model = ob_model_module.operation.get_asm().replace(","," ")
    origin_model = origin_model.replace("\n"," EOVector")
    tempRow = [ob_model, origin_model]
    writer.writerow(tempRow)
csv_file.close()

torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    ob_model_module,
    "builtin.module(func.func(torch-insert-conv))",
    "InsertConvs",
)
with open(csv_file1, 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    ob_model = ob_model_module.operation.get_asm().replace(","," ")
    ob_model = ob_model.replace("\n"," EOVector")
    origin_model = ob_model_module.operation.get_asm().replace(","," ")
    origin_model = origin_model.replace("\n"," EOVector")
    tempRow = [ob_model, origin_model]
    writer.writerow(tempRow)
csv_file.close()

torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    ob_model_module,
    "builtin.module(func.func(torch-insert-skip))",
    "InsertSkip",
)
with open(csv_file1, 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    ob_model = ob_model_module.operation.get_asm().replace(","," ")
    ob_model = ob_model.replace("\n"," EOVector")
    origin_model = ob_model_module.operation.get_asm().replace(","," ")
    origin_model = origin_model.replace("\n"," EOVector")
    tempRow = [ob_model, origin_model]
    writer.writerow(tempRow)
csv_file.close()

torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    ob_model_module,
    "builtin.module(func.func(torch-kernel-widening))",
    "KernelWidening",
)
with open(csv_file1, 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    ob_model = ob_model_module.operation.get_asm().replace(","," ")
    ob_model = ob_model.replace("\n"," EOVector")
    origin_model = ob_model_module.operation.get_asm().replace(","," ")
    origin_model = origin_model.replace("\n"," EOVector")
    tempRow = [ob_model, origin_model]
    writer.writerow(tempRow)
csv_file.close()


torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    ob_model_module,
    "builtin.module(func.func(torch-dummy-addition))",
    "DummyAddition",
)
with open(csv_file1, 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    ob_model = ob_model_module.operation.get_asm().replace(","," ")
    ob_model = ob_model.replace("\n"," EOVector")
    origin_model = ob_model_module.operation.get_asm().replace(","," ")
    origin_model = origin_model.replace("\n"," EOVector")
    tempRow = [ob_model, origin_model]
    writer.writerow(tempRow)
csv_file.close()