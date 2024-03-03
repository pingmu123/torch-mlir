# https://github.com/aaron-xichen/pytorch-playground/blob/master/imagenet/alexnet.py

import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

__all__ = ["AlexNet", "alexnet"]


model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


net = AlexNet()
net.eval()
print(net)

# compile to torch mlir
# NCHW layout in pytorch
print("================")
print("origin torch mlir")
print("================")
module = torch_mlir.compile(net, torch.ones(1, 3, 227, 227), output_type="torch")
print(module.operation.get_asm(large_elements_limit=10))


# # # your pass begin

# add your passes here

"""========================DeObfuscation begin========================="""

print(
    "Note: InsertLinearPass may causes the process to be killed due to running for a long time! "
)

Obfuscation = ["Obfuscation Passes:"]
passes = []
np.random.seed(int(time.time()))
randNum = np.random.randint(0, 8)
for i in range(0, 10):
    randNum = np.random.randint(0, 8)
    if randNum % 2 == 0:
        passes.append(1)
        Obfuscation.append("    WidenLayerPass")
        randNum = np.random.randint(0, 8)  # for 'while'
    randNum = np.random.randint(0, 8)
    if randNum % 2 == 0:
        passes.append(1)
        Obfuscation.append("    BranchLayerPass")
        randNum = np.random.randint(0, 8)  # for 'while'
    randNum = np.random.randint(0, 8)  # for 'for'
for i in range(0, 5):  # 5 times
    while randNum % 8:
        passes.append(randNum)
        randNum = np.random.randint(0, 8)
    randNum = np.random.randint(0, 8)  # for 'for'
for randNum in passes:
    if randNum == 1:
        pass
        # Obfuscation.append("    WidenLayerPass")
        # torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        #     module,
        #     "builtin.module(func.func(torch-widen-conv-layer))",
        #     "WidenConvLayer",
        # )
    elif randNum == 2:
        pass
        # Obfuscation.append("    BranchLayerPass")
        # torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        #     module,
        #     "builtin.module(func.func(torch-branch-layer))",
        #     "BranchLayer",
        # )
    elif randNum == 3:
        pass
        # Obfuscation.append("    DummyAdditionPass")
        # torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        #     module,
        #     "builtin.module(func.func(torch-dummy-addition))",
        #     "DummyAddition",
        # )
    elif randNum == 4:
        pass
        Obfuscation.append("    InsertConvsPass")
        torch_mlir.compiler_utils.run_pipeline_with_repro_report(
            module,
            "builtin.module(func.func(torch-insert-conv))",
            "InsertConvs",
        )
    elif randNum == 5:
        print("InsertLinearPass start!")
        Obfuscation.append("    InsertLinearsPass")
        torch_mlir.compiler_utils.run_pipeline_with_repro_report(
            module,
            "builtin.module(func.func(torch-insert-linear))",
            "InsertLinear",
        )
        print("InsertLinearPass has finished!")
    elif randNum == 6:
        pass
        Obfuscation.append("    InsertSkipPass")
        torch_mlir.compiler_utils.run_pipeline_with_repro_report(
            module,
            "builtin.module(func.func(torch-insert-skip))",
            "InsertSkip",
        )
    elif randNum == 7:
        Obfuscation.append("    KernelWideningPass")
        torch_mlir.compiler_utils.run_pipeline_with_repro_report(
            module,
            "builtin.module(func.func(torch-kernel-widening))",
            "KernelWidening",
        )
    else:
        Obfuscation.append("    Jump this Obfuscation")

Obfuscation.append("    DummyAdditionPass")
torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    module,
    "builtin.module(func.func(torch-dummy-addition))",
    "DummyAddition",
)
print("================")
print("after Obfuscations")
print("================")
print(module.operation.get_asm(large_elements_limit=10))

for passName in Obfuscation:
    print(passName)


"""========================Obfuscation end============================="""


"""========================DeObfuscation begin========================="""

# Others passes
# DeObfuscation times = DeObfuscationPassNum + 1(get a loop) - 1(Skip)
DeObfTimes = 7
for i in range(0, 1):
    pass
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(func.func(torch-anti-insert-skip))",
        "AntiInsertSkip",
    )

for i in range(0, DeObfTimes):
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(func.func(torch-anti-widen-conv-layer))",
        "",
    )
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(func.func(torch-anti-branch-layer))",
        "AntiBranchLayer",
    )
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(func.func(torch-anti-dummy-addition))",
        "AntiDummyAddition",
    )
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(func.func(torch-anti-insert-conv))",
        "AntiInsertConvs",
    )
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(func.func(torch-anti-insert-linear))",
        "AntiInsertLinear",
    )
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(func.func(torch-anti-insert-skip))",
        "AntiInsertSkip",
    )
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(func.func(torch-anti-kernel-widening))",
        "AntiAntiKernelWidening",
    )

"""========================Obfuscation end============================="""

# # # your pass end
print("====================")
print("after DeObfuscations")
print("====================")
print(module.operation.get_asm(large_elements_limit=10))

# lowering and run
print("=====================")
print("after lower to linalg")
print("=====================")
torch_mlir.compiler_utils.run_pipeline_with_repro_report(
    module,
    "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
    "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
)
print(module.operation.get_asm(large_elements_limit=10))


# print("================")
# print("run model")
# print("================")
# backend = refbackend.RefBackendLinalgOnTensorsBackend()
# compiled = backend.compile(module)
# jit_module = backend.load(compiled)
# jit_func = jit_module.forward
# out1 = jit_func(torch.ones(1, 3, 227, 227).numpy())
# out2 = jit_func(torch.zeros(1, 3, 227, 227).numpy())
# print("output:")
# print(out1)
# print(out2)


# # origin
# module_origin = torch_mlir.compile(
#     net, torch.ones(1, 3, 227, 227), output_type="linalg-on-tensors"
# )
# jit_func_origin = backend.load(backend.compile(module_origin)).forward
# out1_origin = jit_func_origin(torch.ones(1, 3, 227, 227).numpy())
# out2_origin = jit_func_origin(torch.zeros(1, 3, 227, 227).numpy())
# print("origin output:")
# print(out1_origin)
# print(out2_origin)


# compare
# print("diffs:")
# print(out1 - out1_origin)
# print(out2 - out2_origin)


# def alexnet(pretrained=False, model_root=None, **kwargs):
#     model = AlexNet(**kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls["alexnet"], model_root))
#     return model
