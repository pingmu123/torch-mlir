""""
提取主要信息：

    1. 读取data.csv中的数据

    2. 分词同时过滤掉不需要的词汇

    3.写入token.csv中

"""

import csv

csv.field_size_limit(2**30)  # default: 131072
import torch

# file path
csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
csv_file1 = csv_file_path + "data.csv"
csv_file2_origin = csv_file_path + "token_origin.csv"
csv_file2_no_ob_info = csv_file_path + "token_no_ob_info.csv"
csv_file2_with_ob_info = csv_file_path + "token_with_ob_info.csv"

csv_file_header = ["ob_models_input", "origin_models_label"]


# special symbol
model_token_padding = "<PAD>"  # padding
model_token_beginning = "<BOS>"  # begin symbol
model_token_ending = "<EOS>"  # end symbol
ob_flag_info = "ob_flag_info"

# with open(csv_file2_no_ob_info, "w", newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(csv_file_header)
# csv_file.close()

# with open(csv_file2_with_ob_info, "w", newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(csv_file_header)
# csv_file.close()

# with open(csv_file2_origin, "w", newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(csv_file_header)
# csv_file.close()

csv_reader = csv.reader(open(csv_file1, encoding="utf-8"))
# next(csv_reader)  # jump headLine

# i = 0
# for row in csv_reader:
#     print("length of ", i, ": ", len(row[0]))
#     i += 1
#     if i == 100:
#         exit()


def getSNdict(l: list) -> dict:
    SNdict = {}
    SN = 0
    for token in l:
        # print(token)
        if token[0] == "%":
            if token not in SNdict:
                SNdict[token] = SN
                SN = SN + 1
            else:
                pass
        else:
            pass
    return SNdict


def str2token(s: str) -> list:
    tokens = []
    # pre processing

    # 1. special: input %arg0 shape
    pos = s.find(":")
    s1 = s[0:pos]
    s2 = s[pos + 1 : len(s)]
    s = s1 + " to" + s2  # ' to' not 'to'
    pos1 = s.find("->")
    pos2 = s.find("EOOperation")
    s = s[0:pos2] + s[pos2 + len("EOOperation") : len(s)]
    pos2 = s.find("EOOperation")  # second EOOperation
    s1 = s[0:pos1]
    s2 = s[pos2 : len(s)]
    s = s1 + " " + s2

    # 2. final ':'
    pos = s.rfind(":")
    s1 = s[0:pos]
    s2 = s[pos + 1 : len(s)]
    s = s1 + "to" + s2
    while s.find(":") != -1:
        begin = s.find(":")
        # NOTES:
        #   if end1 = s[begin+1:len(s)].find('->')
        #   then end1 from 0 to ...
        end1 = (
            begin + 1 + s[begin + 1 : len(s)].find("->")
        )  # +1: find in the behind of ':'
        end2 = (
            begin + 1 + s[begin + 1 : len(s)].find(":")
        )  # for torch.tensor.literal and ...
        if end1 != begin or end2 != begin:
            if (end1 != begin and end2 != begin and end1 < end2) or (
                end1 != begin and end2 == begin
            ):
                s1 = s[0:begin]
                s2 = s[end1 : len(s)]
                s = (
                    s1 + s2
                )  # delete tokens which in middle of ':' and '->', include ':'
            else:  # (end1 != begin and end2 != begin and end2 < end1) or (end1 == begin and end2 != begin):
                s1 = s[0:begin]
                s2 = s[end2 + 1 : len(s)]
                s = (
                    s1 + "to" + s2
                )  # delete tokens which in middle of ':' and ':', include ':'
    s = s.replace("->", " to")
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    s = s.replace("<", " ")
    s = s.replace(">", " ")
    s = s.replace("[", " ")
    s = s.replace("]", " ")
    s = s.replace("{", " ")
    s = s.replace("}", " ")
    s = s.replace("0.000000e+00", "0")
    s = s.replace("1.000000e+00", "1")
    # space * n -> space * 1
    for i in range(0, 5):  # n <= 6
        s = s.replace("  ", " ")

    s = s.split(" ")
    pre_token = ""
    for token in s:
        if len(token) > 0:
            if token[0] == "%":
                tokens.append(token)
            elif token[0] == "!":
                if token == "!torch.vtensor":
                    if pre_token == "to":
                        tokens.append("torch.vtensor")
            elif token[0] == "t":
                if (
                    token == "torch.debug_module_name"
                    or token == "tensor"
                    or token == "to"
                ):
                    pass
                elif token == "true":
                    tokens.append("1")
                else:
                    tokens.append(token)
            elif (token[0] >= "0" and token[0] <= "9") or token[0] == "-":
                if token.find("xf32") == -1:  # not include 'xf32'
                    tokens.append(token)
                else:
                    pass
            elif token == "return":
                tokens.append(token)
            elif token[0] == '"':
                token = token.replace('"', "")
                if token.find("0x") == 0:
                    tokens.append(token)
                else:
                    pass
            elif token == "EOOperation":
                if pre_token != "EOOperation":
                    tokens.append(token)
            elif token == "false":
                tokens.append("0")
            else:
                pass
        pre_token = token
    SNdict = getSNdict(tokens)
    for i in range(0, len(tokens)):
        if tokens[i] in SNdict:
            tokens[i] = "Op" + str(SNdict[tokens[i]])
    return tokens


# delete parameters
def get_simple_model_token(tokens: list) -> list:
    res = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "torch.vtensor.literal":
            res.append(tokens[i])
            while i < len(tokens) and tokens[i] != "torch.vtensor":
                i = i + 1
        else:
            res.append(tokens[i])
            i = i + 1
            # if tokens[i] != 'EOOperation':
            #     res.append(tokens[i])
            #     i = i + 1
            # else: i += 1
    return res


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


def check_tensor_in_list(tensor, tensor_list):
    for t in tensor_list:
        if torch.equal(tensor, t):
            return True
    return False


def get_origin_model_tensors(tokens: list) -> list:
    tensors = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "torch.vtensor.literal":
            if (
                i + 1 < len(tokens)
                and len(tokens[i + 1]) > 2
                and tokens[i + 1][0:2] == "0x"
            ):
                i += 1  # s[i] = "0x...."
                tensor = hex2float(tokens[i])
                i += 2  # jump "torch.vtensor", get to shape_info
                shape_info = []
                while tokens[i] != "EOOperation":
                    shape_info.append(int(tokens[i]))
                    i += 1
                tensor = torch.tensor(tensor).reshape(shape_info)
                tensors.append(tensor)

            else:
                i += 1  # jump 'torch.vtensor.literal', get to parameters
                tensor = []
                while tokens[i] != "torch.vtensor":
                    tensor.append(float(tokens[i]))
                    i += 1
                i += 1  # jump "torch.vtensor", get to shape_info
                size = 1
                shape_info = []
                while tokens[i] != "EOOperation":
                    shape_info.append(int(tokens[i]))
                    size *= int(tokens[i])
                    i += 1
                while len(tensor) < size:  # solve all params are equal in torch-mlir
                    tensor.append(tensor[0])
                tensor = torch.tensor(tensor).reshape(shape_info)
                tensors.append(tensor)
        i += 1

    return tensors


def get_simple_model_token_with_ob_info(tokens: list, tensors: list) -> list:
    res = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "torch.vtensor.literal":
            res.append(tokens[i])
            if (
                i + 1 < len(tokens)
                and len(tokens[i + 1]) > 2
                and tokens[i + 1][0:2] == "0x"
            ):
                i += 1  # s[i] = "0x...."
                tensor = hex2float(tokens[i])
                i += 2  # jump "torch.vtensor", get to shape_info
                shape_info = []
                while tokens[i] != "EOOperation":
                    shape_info.append(int(tokens[i]))
                    i += 1
                tensor = torch.tensor(tensor).reshape(shape_info)
                res.append("torch.vtensor")
                for tensor_shape in shape_info:
                    res.append(str(tensor_shape))
                if check_tensor_in_list(tensor, tensors) == False:
                    res.append(ob_flag_info)
            else:
                i += 1  # jump 'torch.vtensor.literal', get to parameters
                tensor = []
                while tokens[i] != "torch.vtensor":
                    tensor.append(float(tokens[i]))
                    i += 1
                i += 1  # jump "torch.vtensor", get to shape_info
                size = 1
                shape_info = []
                while tokens[i] != "EOOperation":
                    shape_info.append(int(tokens[i]))
                    size *= int(tokens[i])
                    i += 1
                while len(tensor) < size:  # solve all params are equal in torch-mlir
                    tensor.append(tensor[0])
                tensor = torch.tensor(tensor).reshape(shape_info)
                res.append("torch.vtensor")
                for tensor_shape in shape_info:
                    res.append(str(tensor_shape))
                if check_tensor_in_list(tensor, tensors) == False:
                    res.append(ob_flag_info)
            res.append("EOOperation")
        else:
            res.append(tokens[i])
        i += 1

    return res


count = 0
simple_ob_model_token = []
for row in csv_reader:
    count += 1
    print("model:", count)
    if count % 2 == 1:
        ob_model = row[0]
        ob_model_token = str2token(ob_model)
        # ob_model_token.insert(0, model_token_beginning)
        # ob_model_token.append(model_token_ending)
        # simple_ob_model_token = get_simple_model_token(ob_model_token)
    else:
        origin_model = row[0]
        origin_model_token = str2token(origin_model)
        # origin_model_token.insert(0, model_token_beginning)
        # origin_model_token.append(model_token_ending)
        # simple_origin_model_token = get_simple_model_token(origin_model_token)

        # tensors = get_origin_model_tensors(origin_model_token)
        # simple_ob_model_token_with_ob_info = get_simple_model_token_with_ob_info(
        #     ob_model_token, tensors
        # )

        # write to token.csv
        tempRow = [ob_model_token, origin_model_token]
        with open(csv_file2_origin, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(tempRow)
            tempRow.clear()
        csv_file.close()

        # tempRow = [simple_ob_model_token, simple_origin_model_token]
        # with open(csv_file2_no_ob_info, "a", newline="") as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerow(tempRow)
        # csv_file.close()

        # tempRow = [simple_ob_model_token_with_ob_info, simple_origin_model_token]
        # with open(csv_file2_with_ob_info, "a", newline="") as csv_file:
        #     writer = csv.writer(csv_file)
        #     writer.writerow(tempRow)
        # csv_file.close()
