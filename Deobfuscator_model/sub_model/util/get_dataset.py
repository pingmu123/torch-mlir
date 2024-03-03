"""
    数据集：
    
        1. 从token_origin.csv中提取出'torch.vtensor.literal' Op

        2. 对比混淆前后的Op, 新产生的Op为混淆Op(直接使用该Op的Op也往往是混淆Op)

        3. 将所有Op放入dataset.csv文件并打上Label

"""

import numpy as np
import csv
import torch

csv.field_size_limit(2**30)  # default: 131072
csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
# csv_file1 = csv_file_path + "data.csv"
# csv_file2 = csv_file_path + "token.csv"
# csv_file3 = csv_file_path + "token2vec.csv"
csv_file_token = csv_file_path + "token_origin.csv"
csv_file_dataset = (
    "/home/pingmu123/torch-mlir/Deobfuscator_model/sub_model/dataset/dataset.csv"
)

# padding_shape = (3, 224, 224)
padding = -100


def hex2float(s: str) -> list:
    float_list = []
    hex2bin = {
        "0": "0000",
        "1": "0001",
        "2": "0010",
        "3": "0011",
        "4": "0100",
        "5": "0101",
        "6": "0110",
        "7": "0111",
        "8": "1000",
        "9": "1001",
        "A": "1010",
        "B": "1011",
        "C": "1100",
        "D": "1101",
        "E": "1110",
        "F": "1111",
    }
    s = s[2 : len(s)]
    times = len(s) // 8
    for i in range(0, times):
        hex = s[i * 8 : i * 8 + 8]
        hex = hex[6:8] + hex[4:6] + hex[2:4] + hex[0:2]  # large end -> little end
        bin = ""
        for c in hex:
            bin = bin + hex2bin[c]
        sign = bin[0]
        e = -127
        sum = 128
        for j in range(1, 9):
            num = 0 if bin[j] == "0" else 1
            e = e + num * sum
            sum = sum // 2
        t = 1.0
        sum_f = 0.5
        for j in range(9, 32):
            num = 0 if bin[j] == "0" else 1
            t = t + num * sum_f
            sum_f = sum_f / 2
        fRes = t * (2**e)
        if sign == "1":
            fRes = -fRes
        if abs(fRes) < 10e-38:  # approach to 0 then 0
            fRes = 0.0
        float_list.append(fRes)

    return float_list


def get_models_tensor_list(csv_reader):
    models = []
    shapes = []
    for row in csv_reader:
        ob_tensor_list = []
        origin_tensor_list = []
        # row: ob_model_token, origin_model_token
        s = row[0]
        s = s.replace("[", "")
        s = s.replace("]", "")
        s = s.replace("'", "")
        s = s.replace(" ", "")  # delete space
        s = s.split(",")
        for i in range(0, len(s)):
            if s[i] == "torch.vtensor.literal":
                if i + 1 < len(s) and len(s[i + 1]) > 2 and s[i + 1][0:2] == "0x":
                    i += 1  # s[i] = "0x...."
                    tensor = hex2float(s[i])
                    i += 2  # jump "torch.vtensor", get to shape_info
                    shape_info = []
                    while s[i] != "EOOperation":
                        shape_info.append(int(s[i]))
                        i += 1
                    tensor = torch.tensor(tensor).reshape(shape_info)
                    ob_tensor_list.append(tensor)

                else:
                    i += 1  # jump 'torch.vtensor.literal', get to parameters
                    tensor = []
                    while s[i] != "torch.vtensor":
                        tensor.append(float(s[i]))
                        i += 1
                    i += 1  # jump "torch.vtensor", get to shape_info
                    size = 1
                    shape_info = []
                    while s[i] != "EOOperation":
                        shape_info.append(int(s[i]))
                        size *= int(s[i])
                        i += 1
                    while (
                        len(tensor) < size
                    ):  # solve all params are equal in torch-mlir
                        tensor.append(tensor[0])
                    tensor = torch.tensor(tensor).reshape(shape_info)
                    ob_tensor_list.append(tensor)
        s = row[1]
        s = s.replace("[", "")
        s = s.replace("]", "")
        s = s.replace("'", "")
        s = s.replace(" ", "")  # delete space
        s = s.split(",")
        for i in range(0, len(s)):
            if s[i] == "torch.vtensor.literal":
                if i + 1 < len(s) and len(s[i + 1]) > 2 and s[i + 1][0:2] == "0x":
                    i += 1  # s[i] = "0x...."
                    tensor = hex2float(s[i])
                    i += 2  # jump "torch.vtensor", get to shape_info
                    shape_info = []
                    while s[i] != "EOOperation":
                        shape_info.append(int(s[i]))
                        i += 1
                    tensor = torch.tensor(tensor).reshape(shape_info)
                    origin_tensor_list.append(tensor)

                else:
                    i += 1  # jump 'torch.vtensor.literal', get to parameters
                    tensor = []
                    while s[i] != "torch.vtensor":
                        tensor.append(float(s[i]))
                        i += 1
                    i += 1  # jump "torch.vtensor", get to shape_info
                    size = 1
                    shape_info = []
                    while s[i] != "EOOperation":
                        shape_info.append(int(s[i]))
                        size *= int(s[i])
                        i += 1
                    while (
                        len(tensor) < size
                    ):  # solve all params are equal in torch-mlir
                        tensor.append(tensor[0])
                    tensor = torch.tensor(tensor).reshape(shape_info)
                    origin_tensor_list.append(tensor)

        models.append([ob_tensor_list, origin_tensor_list])

    return models, shapes


csv_reader = csv.reader(open(csv_file_token, encoding="utf-8"))
next(csv_reader)  # jump headLine
print("get tensor begin")
models_tensor_list, models_tensor_shape_list = get_models_tensor_list(csv_reader)
print("get tensor end")


# with open(csv_file_dataset, "w", newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(["tensor", "label"])
# csv_file.close()


def check_tensor_in_list(tensor, tensor_list):
    for t in tensor_list:
        if torch.equal(tensor, t):
            return True
    return False


def make_dataset(models_tensor_list) -> None:
    # dataset:
    #   tensor, shape, label
    data_all = []
    count = 0
    for model_tensor_list in models_tensor_list:
        ob_model_tensor_list, origin_model_tensor_list = model_tensor_list
        print("length:", len(ob_model_tensor_list))
        for tensor in ob_model_tensor_list:
            print("tensor shape:", tensor.shape)
            label = 1
            if check_tensor_in_list(tensor, origin_model_tensor_list):
                label = 0  # not ob_tensor
            data = []
            shape = list(tensor.shape)
            size = 1
            for num in shape:
                size *= num
            if size > 3 * 224 * 224:
                print(size, "input overflow!")
                continue
            if size == 1:
                continue  # don't care
            t = torch.zeros(3 * 224 * 224)
            tensor = tensor.contiguous().view(-1)
            t[0:size] = tensor[0:size]
            t[size:] = padding
            data.append(t.tolist())
            data.append(label)
            data_all.append(data)
            count += 1
            if (count % 5) == 0:  # solve killed
                # write to dataset.csv
                with open(csv_file_dataset, "a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    for data in data_all:
                        writer.writerow(data)
                csv_file.close()
                data_all.clear()


# make dataset
print("make dataset begin")
make_dataset(models_tensor_list)
print("make dataset end")
