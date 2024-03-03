"""
生成数据集：

    1. 随机生成足够多的DNN模型

    2. 将上述DNN模型转化为torch.mlir表示

    3. 由上述torch.mlir表示得到混淆后的torch.mlir表示

    4. 将3和2分别作为样本的data和label得到数据集并存入data.csv文件中

"""

import csv
import copy

csv.field_size_limit(2**30)  # default: 131072
import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

# from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
csv_file1 = csv_file_path + "data.csv"

# csv_file_header = ["ob_models_input", "origin_models_label"]
# with open(csv_file1, "w", newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(csv_file_header)
# csv_file.close()

np.random.seed(int(time.time()))

num_of_models = 20

# Obfuscations Combined Method
"""
total: 7 obfuscation transformations.
    dict: num -> transformation
        0 -> widenLayer
        1 -> branchLayer
        2 -> insertConv(s)
        3 -> insertSkip
        4 -> dummyAddition
        5 -> kernelWidening
        6 -> insertLinear
"""


def get_trans_list(num: int) -> list:
    """
    get obfuscation transformations list.
    param:
        num: obfauscation count
        return: ob_trans_list
    """

    ob_trans = []
    ob_trans.append(0)  # special: widenLayer
    for _ in range(0, num):
        for _ in range(0, 6):
            randNum = np.random.randint(0, 6) + 1
            add = np.random.randint(0, 2)  # 50% do transformation
            if add % 2:
                ob_trans.append(randNum)

    return ob_trans


def do_transformation(ob_trans: list, model_module) -> None:
    """
    Do obfuscation transformation in ob_trans_list and write result to data.csv.
    param:
        ob_trans: obfuscation transformation list
        module_module: model module in torch-mlir
    """

    labels = []
    # origin_model
    temp_origin_model = model_module.operation.get_asm().replace(",", " ")
    temp_origin_model = temp_origin_model.replace("\n", " EOOperation")
    origin_model = copy.deepcopy(temp_origin_model)
    labels.append(origin_model)

    # module = copy.deepcopy(model_module) # TypeError: cannot pickle 'torch_mlir._mlir_libs._mlir.ir.Module' object
    for ob_pass in ob_trans:
        if ob_pass == 0:
            torch_mlir.compiler_utils.run_pipeline_with_repro_report(
                model_module,
                "builtin.module(func.func(torch-widen-conv-layer))",
                "WidenConvLayer",
            )
            ob_model_tmep = (
                model_module.operation.get_asm()
                .replace(",", " ")
                .replace("\n", " EOOperation")
            )
            if len(ob_model_tmep) > 250000000:
                break
            # if len(ob_model_tmep) == len(origin_model):  # no ob in this loop
            #     continue
            origin_model = copy.deepcopy(ob_model_tmep)
            labels.append(origin_model)
        if ob_pass == 1:
            torch_mlir.compiler_utils.run_pipeline_with_repro_report(
                model_module,
                "builtin.module(func.func(torch-branch-layer))",
                "BranchLayer",
            )
            ob_model_tmep = (
                model_module.operation.get_asm()
                .replace(",", " ")
                .replace("\n", " EOOperation")
            )
            if len(ob_model_tmep) > 250000000:
                break
            if len(ob_model_tmep) == len(origin_model):  # no ob in this loop
                continue
            origin_model = copy.deepcopy(ob_model_tmep)
            labels.append(origin_model)
        elif ob_pass == 2:
            torch_mlir.compiler_utils.run_pipeline_with_repro_report(
                model_module,
                "builtin.module(func.func(torch-insert-conv))",
                "InsertConvs",
            )
            ob_model_tmep = (
                model_module.operation.get_asm()
                .replace(",", " ")
                .replace("\n", " EOOperation")
            )
            if len(ob_model_tmep) > 250000000:
                break
            if len(ob_model_tmep) == len(origin_model):
                continue
            origin_model = copy.deepcopy(ob_model_tmep)
            labels.append(origin_model)
        elif ob_pass == 3:
            torch_mlir.compiler_utils.run_pipeline_with_repro_report(
                model_module,
                "builtin.module(func.func(torch-insert-skip))",
                "InsertSkip",
            )
            ob_model_tmep = (
                model_module.operation.get_asm()
                .replace(",", " ")
                .replace("\n", " EOOperation")
            )
            if len(ob_model_tmep) > 250000000:
                break
            if len(ob_model_tmep) == len(origin_model):
                continue
            origin_model = copy.deepcopy(ob_model_tmep)
            labels.append(origin_model)
        elif ob_pass == 4:
            torch_mlir.compiler_utils.run_pipeline_with_repro_report(
                model_module,
                "builtin.module(func.func(torch-dummy-addition))",
                "DummyAddition",
            )
            ob_model_tmep = (
                model_module.operation.get_asm()
                .replace(",", " ")
                .replace("\n", " EOOperation")
            )
            if len(ob_model_tmep) > 250000000:
                break
            if len(ob_model_tmep) == len(origin_model):
                continue
            origin_model = copy.deepcopy(ob_model_tmep)
            labels.append(origin_model)
        elif ob_pass == 5:
            torch_mlir.compiler_utils.run_pipeline_with_repro_report(
                model_module,
                "builtin.module(func.func(torch-kernel-widening))",
                "KernelWidening",
            )
            ob_model_tmep = (
                model_module.operation.get_asm()
                .replace(",", " ")
                .replace("\n", " EOOperation")
            )
            if len(ob_model_tmep) > 250000000:
                break
            if len(ob_model_tmep) == len(origin_model):
                continue
            origin_model = copy.deepcopy(ob_model_tmep)
            labels.append(origin_model)
        else:  # ob_pass == 6
            # ob_model_tmep = (
            #     model_module.operation.get_asm()
            #     .replace(",", " ")
            #     .replace("\n", " EOOperation")
            # )
            # if len(ob_model_tmep) > 250000000:
            #     break
            # origin_model = copy.deepcopy(ob_model_tmep)
            # labels.append(origin_model)
            pass
            # now we jump this pass
            # torch_mlir.compiler_utils.run_pipeline_with_repro_report(
            #     model_module,
            #     "builtin.module(func.func(torch-insert-linear))",
            #     "InsertLinear",
            # )

    # ob_model
    print("ob finished.")
    ob_model_final = (
        model_module.operation.get_asm().replace(",", " ").replace("\n", " EOOperation")
    )
    if ob_trans[len(ob_trans) - 1] != 3:
        torch_mlir.compiler_utils.run_pipeline_with_repro_report(
            model_module,
            "builtin.module(func.func(torch-anti-insert-skip))",
            "AntiInsertSkip",
        )
        ob_model_tmep = (
            model_module.operation.get_asm()
            .replace(",", " ")
            .replace("\n", " EOOperation")
        )
        if len(ob_model_tmep) <= 250000000 and len(ob_model_tmep) != len(origin_model):
            origin_model = copy.deepcopy(ob_model_tmep)
            labels.append(origin_model)

    with open(csv_file1, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # for label in labels:
        #     writer.writerow([ob_model_final])
        #     writer.writerow([label])

        # for get_ob_info dataset
        writer.writerow([ob_model_final])
        writer.writerow([labels[0]])
    csv_file.close()


# the sub_model or full_model of LeNet-5, AlexNet, VGG, GoogleNet, ResNet, ...
"""
    We randomly change these model's structure to make our dataset. 
"""


class conv_layer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        conv_kernel_size,
        conv_stride,
        pool_kernel_size,
        pool_stride,
    ):
        super(conv_layer, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, conv_kernel_size, conv_stride)
        pool_type = np.random.randint(0, 2) % 2
        if pool_type == 1:
            self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
        else:
            self.pool = nn.AvgPool2d(pool_kernel_size, pool_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = F.relu(x)

        return x


class fc_layer(nn.Module):
    def __init__(self, in_feature, out_feature, bias):
        super(fc_layer, self).__init__()

        self.linear = nn.Linear(in_feature, out_feature, bias)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x


class inception_layer(nn.Module):
    """
    refer to https://github.com/Lornatang/GoogLeNet-PyTorch/blob/main/model.py
    """

    def __init__(
        self, in_channel, out_ch1, out_ch3_1, out_ch3_2, out_ch5_1, out_ch5_2, pool_proj
    ):
        super(inception_layer, self).__init__()

        # branch1
        self.conv1 = nn.Conv2d(in_channel, out_ch1, 1)
        self.bn1 = nn.BatchNorm2d(out_ch1, eps=0.001)

        # branch2
        self.conv2_1 = nn.Conv2d(in_channel, out_ch3_1, 1)
        self.bn2_1 = nn.BatchNorm2d(out_ch3_1, eps=0.001)
        self.conv2_2 = nn.Conv2d(out_ch3_1, out_ch3_2, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(out_ch3_2, eps=0.001)

        # branch3
        self.conv3_1 = nn.Conv2d(in_channel, out_ch5_1, 1)
        self.bn3_1 = nn.BatchNorm2d(out_ch5_1, eps=0.001)
        self.conv3_2 = nn.Conv2d(out_ch5_1, out_ch5_2, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(out_ch5_2, eps=0.001)

        # branch4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channel, pool_proj, 1)
        self.bn4 = nn.BatchNorm2d(pool_proj, eps=0.001)

        self.relu = nn.ReLU()

    def forward(self, x):
        branch1 = self.conv1(x)
        branch1 = self.bn1(branch1)
        branch1 = self.relu(branch1)

        branch2 = self.conv2_1(x)
        branch2 = self.bn2_1(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.conv2_2(branch2)
        branch2 = self.bn2_2(branch2)
        branch2 = self.relu(branch2)

        branch3 = self.conv3_1(x)
        branch3 = self.bn3_1(branch3)
        branch3 = self.relu(branch3)
        branch3 = self.conv3_2(branch3)
        branch3 = self.bn3_2(branch3)
        branch3 = self.relu(branch3)

        branch4 = self.maxpool(x)
        branch4 = self.conv4(branch4)
        branch4 = self.bn4(branch4)
        branch4 = self.relu(branch4)

        out = [branch1, branch2, branch3, branch4]
        out = torch.cat(out, 1)

        return out


class residual_block(nn.Module):
    """
    refer to https://github.com/ZOMIN28/ResNet18_Cifar10_95.46/blob/main/utils/ResNet.py
    """

    def __init__(self, in_channel, out_channel, conv_kernel_size, padding_size):
        super(residual_block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channel, out_channel, conv_kernel_size, padding=padding_size
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        # out_channel of second conv is equal to first conv
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, conv_kernel_size, padding=padding_size
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class random_DNN_model(nn.Module):
    """
    Generate random DNN models.
    params:
        layer_num: depth of DNN
        layer_type:
            0: conv + relu + pool
            1: inception
            2: residual block
            /: fc
    """

    def __init__(self, n, c, h, w):
        super(random_DNN_model, self).__init__()
        #  we assume that h is always equal to w
        in_ch = c
        in_h = h
        layers = []
        layer_num = (
            np.random.randint(0, 3) % 3 + 1
        )  # we generate DNNs that have 1~3 block(s) + n * fc_layer(n = 1, 2)
        layer_types = [
            0,
            0,
            1 if np.random.randint(0, 2) % 2 else 2,
        ]  # conv*2 + inception or residual block

        for layer_type in layer_types:
            if in_h <= 22:
                break  # its size must bigger than ker_size(max 11 but we assign 11*2 for conv and pooling) and pool_ker_size
            if layer_type == 0:
                print("build conv_layer begin")
                out_ch = 2 ** (np.random.randint(0, 4) % 4 + 3)  # 8, 16, 32, 64
                ker_size = np.random.randint(0, 5) % 5 * 2 + 3  # 3, 5, 7, 9, 11
                # both h and w of input_size are small, so stride and pool_size set small too
                conv_stride = np.random.randint(0, 1) % ker_size + 1  # 1
                pool_ker_size = np.random.randint(0, 1) % 1 + 2  # 2
                # pool_stride = np.random.randint(0, pool_ker_size) % pool_ker_size + 1
                pool_stride = pool_ker_size
                layers.append(
                    conv_layer(
                        in_ch, out_ch, ker_size, conv_stride, pool_ker_size, pool_stride
                    )
                )
                # update input of next layer
                in_ch = out_ch
                in_h = math.floor((in_h - ker_size) / conv_stride) + 1  # after conv
                in_h = (
                    math.floor((in_h - pool_ker_size) / pool_stride)
                    + 1  # Note: floor too
                )  # after pool
                print("build conv_layer end")
            elif layer_type == 1:
                print("build inception_layer begin")
                out_ch1 = 2 ** (np.random.randint(0, 3) % 3 + 3)  # 8, 16, 32
                out_ch3_1 = 2 ** (np.random.randint(0, 3) % 3 + 3)
                out_ch3_2 = 2 ** (np.random.randint(0, 3) % 3 + 3)
                out_ch5_1 = 2 ** (np.random.randint(0, 3) % 3 + 3)
                out_ch5_2 = 2 ** (np.random.randint(0, 3) % 3 + 3)
                pool_proj = 2 ** (np.random.randint(0, 3) % 3 + 3)
                layers.append(
                    inception_layer(
                        in_ch,
                        out_ch1,
                        out_ch3_1,
                        out_ch3_2,
                        out_ch5_1,
                        out_ch5_2,
                        pool_proj,
                    )
                )
                in_ch = out_ch1 + out_ch3_2 + out_ch5_2 + pool_proj
                print("build inception_layer end")
            else:
                print("build residual_block begin")
                # out_ch = 2 ** (np.random.randint(0, 4) % 6 + 3)  # 8, 16, 32, 64
                out_ch = in_ch  # x + skip(x)
                ker_size = np.random.randint(0, 6) % 6 * 2 + 1  # 1, 3, 5, 7, 9, 11
                # stride is 1
                padding_size = (ker_size - 1) // 2
                layers.append(residual_block(in_ch, out_ch, ker_size, padding_size))
                # update: only out_ch
                in_ch = out_ch
                print("build residual_block end")

        # add fc_layers: it would change tensor's shape size, so final use it.
        print("build fc_layer begin")
        num_of_fc_layers = 1 if np.random.randint(0, 2) % 2 else 2
        print("num_of_fc_layers:", num_of_fc_layers)
        for i in range(0, num_of_fc_layers):
            if i == 0:
                try:
                    out_feature = 2 ** (
                        np.random.randint(0, 5) % 5 + 3
                    )  # 8, 16, 32, 64, 128
                    bias = True if (np.random.randint(0, 2) % 2) == 0 else False
                    print(in_ch, in_h)
                    layers.append(
                        fc_layer(in_ch * in_h * in_h, out_feature, bias)
                    )  # i == 0
                except:
                    continue
                # update
                in_ch = out_feature
            else:  # i > 0
                try:
                    out_feature = 2 ** (
                        np.random.randint(0, 5) % 5 + 3
                    )  # 8, 16, 32, 64, 128
                    bias = True if (np.random.randint(0, 2) % 2) == 0 else False
                    layers.append(fc_layer(in_ch, out_feature, bias))  # i == 0
                except:
                    continue
                # update
                in_ch = out_feature

        print("build fc_layer end")

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # input_size: [n, c, h, w]
        for layer in self.layers:
            x = layer(x)

        return x


# begin
print("Now start to build dataset!")


def init_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


# input size:
#    n, c, h, w
error_model_count = 0
for i_0 in range(0, num_of_models):
    n = 1
    c = 1 if np.random.randint(0, 2) == 0 else 3  # c = 1 or 3
    # h = 224 if np.random.randint(0, 2) == 0 else 299 # h = 224 or 299
    h = 224
    w = h
    try:
        model = random_DNN_model(n, c, h, w)
    except:
        error_model_count += 1
        print(f"Generate DNN model failed! total: {error_model_count} times\n")
        continue
    model.apply(init_weights)
    model.eval()

    # test model run correctly
    """
    module = torch_mlir.compile(model, torch.ones(n, c, h, w), output_type="torch")
    print(module.operation.get_asm(large_elements_limit=10))
    torch_mlir.compiler_utils.run_pipeline_with_repro_report(
        module,
        "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline)",
        "Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
    )
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(module)
    jit_module = backend.load(compiled)
    jit_func = jit_module.forward
    out1 = jit_func(torch.ones(n, c, h, w).numpy())
    out2 = jit_func(torch.zeros(n, c, h, w).numpy())
    print("output:")
    print(out1)
    print(out2)
    exit()
    """

    # num = 3 # todo: give an scope
    ob_records = []
    num_of_ob_round_max = 3
    error_trans_count = 0
    # for i_1 in range(0, 2):  # 1 model, 2 different obfuscations
    for i_1 in range(0, 1):  # for get_ob_info
        ob_trans = get_trans_list(np.random.randint(1, num_of_ob_round_max) + 1)
        ob_records.append(ob_trans)
        # if obfuscations of model is erroneous, we will give obfuscations up.
        # And ob_model == origin_model at this time.
        try:
            print("ob begin")
            module = torch_mlir.compile(
                model, torch.ones(n, c, h, w), output_type="torch"
            )
            do_transformation(ob_trans=ob_trans, model_module=module)
        except:
            error_trans_count += 1
            print(
                f"Obfuscations of model is erroneous! total: {error_trans_count} times"
            )
            pass

    print("model SN:", i_0 + 1)
    print("step:", round(((i_0 + 1) / num_of_models) * 100, 2), "%")

print("Build dataset has finished!")


# todo: Out of memory
