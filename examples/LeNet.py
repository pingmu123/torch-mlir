import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend


class testNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)

    def forward(self, x):
        # input shape: [1, 1, 28, 28]
        x = self.conv1(x)

        return x


net = testNet()
print(net)

# for name, param in net.fc1.named_parameters():
#     print(f"Parameter name: {name}")
#     print(f"Parameter shape: {param.shape}")
#     print(f"Parameter values: {param}")


# compile to torch mlir
# NCHW layout in pytorch
print("================")
print("origin torch mlir")
print("================")
module = torch_mlir.compile(net, torch.ones(1, 1, 28, 28), output_type="torch")
print(module.operation.get_asm())
csv_file1 = "/home/pingmu123/torch-mlir/myTest/" + "model.csv"
with open(csv_file1, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    # for label in labels:
    #     writer.writerow([ob_model_final])
    #     writer.writerow([label])

    # for get_ob_info dataset
    writer.writerow(
        [module.operation.get_asm().replace(",", " ").replace("\n", " EOOperation")]
    )
csv_file.close()
# print("================")
# print("after WidenConvLayer pass")
# print("================")
# torch_mlir.compiler_utils.run_pipeline_with_repro_report(
#     module,
#     "builtin.module(func.func(torch-widen-conv-layer))",
#     "WidenConvLayer",
# )
# print(module.operation.get_asm(large_elements_limit=10))

# # print("================")
# # print("after InsertSkip pass")
# # print("================")
# # torch_mlir.compiler_utils.run_pipeline_with_repro_report(
# #     module,
# #     "builtin.module(func.func(torch-insert-skip))",
# #     "InsertSkip",
# # )
# # print(module.operation.get_asm(large_elements_limit=10))

# # print("================")
# # print("after InsertConv pass")
# # print("================")
# # torch_mlir.compiler_utils.run_pipeline_with_repro_report(
# #     module,
# #     "builtin.module(func.func(torch-insert-conv{}))",
# #     "InsertSkip",
# # )
# # print(module.operation.get_asm(large_elements_limit=10))

# print("================")
# print("after lower to linalg")
# print("================")
# torch_mlir.compiler_utils.run_pipeline_with_repro_report(
#     module,
#     "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
#     "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
# )
# print(module.operation.get_asm(large_elements_limit=10))

# print("================")
# print("run model")
# print("================")
# backend = refbackend.RefBackendLinalgOnTensorsBackend()
# compiled = backend.compile(module)
# jit_module = backend.load(compiled)
# jit_func = jit_module.forward
# out1 = jit_func(torch.ones(1, 1, 28, 28).numpy())
# out2 = jit_func(torch.zeros(1, 1, 28, 28).numpy())
# print("output:")
# print(out1)
# print(out2)

# module_origin = torch_mlir.compile(
#     net, torch.ones(1, 1, 28, 28), output_type="linalg-on-tensors"
# )
# jit_func_origin = backend.load(backend.compile(module_origin)).forward
# out1_origin = jit_func_origin(torch.ones(1, 1, 28, 28).numpy())
# out2_origin = jit_func_origin(torch.zeros(1, 1, 28, 28).numpy())
# print("origin output:")
# print(out1_origin)
# print(out2_origin)

# print("diffs:")
# print(out1 - out1_origin)
# print(out2 - out2_origin)
