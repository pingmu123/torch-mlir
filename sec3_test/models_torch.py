import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch_mlir
import numpy as np
import time
import copy
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from util import get_sequences, get_SER


# LeNet
class lenet(nn.Module):
    # input shape: [batch_size, 1, 28, 28]
    # output shape: [batch_szie, 10]
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.train(False)

    def forward(self, x):
        # input shape is 1x1x28x28
        # Max pooling over a (2, 2) window, if use default stride, error will happen
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=(2, 2))
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


leNet = lenet()
leNet.eval()

alexNet = models.alexnet()  # 8 Layers
alexNet.eval()

vgg11 = models.vgg11()  # VGG11
vgg11.eval()

squeezeNet1_1 = models.squeezenet1_1()  # squeezeNet1_1: 140 Ops
squeezeNet1_1.eval()

vgg13 = models.vgg13()  # VGG13
vgg13.eval()


mobileNet_v3_small = (
    models.mobilenet_v3_small()
)  # mobileNet_v3_small: 16 Layers, 464 Ops
mobileNet_v3_small.eval()
# vgg16 = models.vgg16()  # VGG16

resNet18 = models.resnet18()  # resNet18: 107 Ops
resNet18.eval()  # Note

vgg19 = models.vgg19()  # VGG19
vgg19.eval()


input_shape_test = (1, 1, 28, 28)
input_shape = (1, 3, 224, 224)

input_tensor = torch.rand(input_shape_test)


distances = []
for _ in range(10):
    # compile to torch mlir
    # NCHW layout in pytorch
    # print("================")
    # print("origin torch mlir")
    # print("================")

    # testNet: leNet
    # module = torch_mlir.compile(leNet, input_tensor, output_type="torch")

    # alexNet, vgg11, squeezeNet1_1, vgg13, mobileNet_v3_small, resNet18, vgg19
    module = torch_mlir.compile(
        alexNet, torch.ones(1, 3, 224, 224), output_type="torch"
    )

    origin_model = module.operation.get_asm()
    # print(module.operation.get_asm(large_elements_limit=10))

    # # # your pass begin

    # add your passes here

    # """========================DeObfuscation begin========================="""

    # print(
    #     "Note: InsertLinearPass may causes the process to be killed due to running for a long time! "
    # )

    Obfuscation = ["Obfuscation Passes:"]
    passes = []
    np.random.seed(int(time.time()))
    randNum = np.random.randint(0, 8)
    for i in range(0, 0):
        randNum = np.random.randint(0, 8)
        if randNum % 2 == 0:
            passes.append(1)
            Obfuscation.append("    WidenLayerPass")
            randNum = np.random.randint(0, 8)  # for 'while'
        randNum = np.random.randint(0, 8)
        if randNum % 2 == 0:
            passes.append(2)
            Obfuscation.append("    BranchLayerPass")
            randNum = np.random.randint(0, 8)  # for 'while'
        randNum = np.random.randint(0, 8)  # for 'for'
    for i in range(0, 3):  # 3 times
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

    # obf_model = module.operation.get_asm()

    # de_obf = module.operation.get_asm()

    # Linalg
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
        "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
    )

    print("================")
    print("run model")
    print("================")
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(module)
    jit_module = backend.load(compiled)
    jit_func = jit_module.forward
    out_obf = jit_func(input_tensor.numpy())  # out_obf: numpy

    module_origin = torch_mlir.compile(
        alexNet, input_tensor, output_type="linalg-on-tensors"
    )
    jit_func_origin = backend.load(backend.compile(module_origin)).forward
    out_origin = jit_func_origin(input_tensor.numpy())

    dis = torch.dist(torch.tensor(out_obf), torch.tensor(out_origin))
    print(dis)
    distances.append(dis.item)

    # origin_seq = get_sequences(origin_model)
    # obf_seq = get_sequences(obf_model)
    # de_obf_seq = get_sequences(de_obf)

    # dist =

with open("distance.txt", "a") as file:
    file.write(str(distances))
