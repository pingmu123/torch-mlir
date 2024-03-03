"""
词嵌入：

    1. 读取token.csv文件

    2. 对 '1' 的token进行词嵌入

    3.写入到token2vec.csv文件喂给model

"""

import sys
sys.path.append('/home/pingmu123/torch-mlir/Deobfuscator_model')
from conf import *
import csv
import copy # you'd better to use deep copy when you want copy a list 

csv.field_size_limit(2**30) # default: 131072
csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
csv_file1 = csv_file_path + "data.csv"
csv_file2 = csv_file_path + "token.csv"
csv_file3 = csv_file_path + "token2vec.csv"

with open(csv_file3, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['ob_models_input', 'origin_models_label'])
csv_file.close()


ob_model_vocab_size = ob_model_vocab_size # 64

def get_ob_model_all_token() -> list:
    tokens = set(())
    csv_reader = csv.reader(open(csv_file2, encoding='utf-8'))
    next(csv_reader) # jump headLine
    for row in csv_reader:
        row[0] = row[0].replace('[','') 
        row[0] = row[0].replace(']','')
        row[0] = row[0].replace("'",'')
        row[0] = row[0].replace(' ','') # delete space
        row[0] = row[0].split(',')
        for token in row[0]:
            if len(token)>0 and ((token[0]>='a' and token[0]<='z') or (token[0]>='A' and token[0]<='Z')):
                tokens.add(token)

    # set中元素顺序是乱的
    words = []
    csv_reader = csv.reader(open(csv_file2, encoding='utf-8'))
    next(csv_reader) # jump headLine
    for row in csv_reader:
        row[0] = row[0].replace('[','') 
        row[0] = row[0].replace(']','')
        row[0] = row[0].replace("'",'')
        row[0] = row[0].replace(' ','') # delete space
        row[0] = row[0].split(',')
        for token in row[0]:
            if token in tokens and token not in words:
                words.append(token)
    return words

def get_origin_model_all_token() -> list:
    tokens = set(())
    csv_reader = csv.reader(open(csv_file2, encoding='utf-8'))
    next(csv_reader) # jump headLine
    for row in csv_reader:
        row[1] = row[1].replace('[','') 
        row[1] = row[1].replace(']','')
        row[1] = row[1].replace("'",'')
        row[1] = row[1].replace(' ','') # delete space
        row[1] = row[1].split(',')
        for token in row[1]:
            tokens.add(token)
    words = []
    csv_reader = csv.reader(open(csv_file2, encoding='utf-8'))
    next(csv_reader) # jump headLine
    for row in csv_reader:
        row[1] = row[1].replace('[','') 
        row[1] = row[1].replace(']','')
        row[1] = row[1].replace("'",'')
        row[1] = row[1].replace(' ','') # delete space
        row[1] = row[1].split(',')
        for token in row[1]:
            if token in tokens and token not in words:
                words.append(token)
    return words

def build_ob_model_vocab(l: list)->dict:
    token2vec={}
    # one-hot embedding
    l.append('torch.preOp.parameters')
    arr_len = len(l)
    for i in range(0, arr_len):
        arr = [0 for _ in range(ob_model_vocab_size)]
        arr[i] = 1
        tup = tuple(arr)
        token2vec[l[i]] = tup
    return token2vec

def build_origin_model_vocab(l: list)->dict:
    token2vec={}
    # one-hot embedding
    arr_len = len(l)
    if(arr_len > origin_model_vocab_size):
        print('arr_len > origin_model_vocab_size: one_hot embedding failed!')
        exit()
    for i in range(0, arr_len):
        arr = [0 for _ in range(origin_model_vocab_size)]
        arr[i] = 1
        tup = tuple(arr)
        token2vec[l[i]] = tup
    return token2vec

def get_vec2token(token2vec: dict) -> dict:
    vec2token={}
    for key, value in token2vec.items():
        vec2token[value] = key
    return vec2token

def build_origin_model_vocab_idx(l: list)->dict: # for computing loss
    stoi={}
    arr_len = len(l) #
    for i in range(0, arr_len):
        stoi[l[i]] = i
    return stoi

def get_itos(d: dict) -> dict:
    itos={}
    for key, value in d.items():
        itos[value] = key
    return itos

# build vocabulary
ob_model_token = get_ob_model_all_token()
ob_model_token2vec = build_ob_model_vocab(ob_model_token)
ob_model_vec2token = get_vec2token(ob_model_token2vec)

origin_model_token = get_origin_model_all_token()
origin_model_token2vec = build_origin_model_vocab(origin_model_token)
origin_model_token2idx = build_origin_model_vocab_idx(origin_model_token)
origin_model_idx2token = get_itos(origin_model_token2idx)
origin_model_vec2token = get_vec2token(origin_model_token2vec)
origin_model_token2vec_size = len(origin_model_token2vec)

inf = inf
INT_MAX = max_len_tgt
topo_pad = -1
parameters_pad = 0
shape_pad = 1

def hex2float(s:str) -> list:
    float_list = []
    hex2bin={
        '0': '0000', '1': '0001', '2': '0010', '3': '0011',
        '4': '0100', '5': '0101', '6': '0110', '7': '0111',
        '8': '1000', '9': '1001', 'A': '1010', 'B': '1011',
        'C': '1100', 'D': '1101', 'E': '1110', 'F': '1111',
        }
    s = s[2:len(s)]
    times = len(s) // 8
    for i in range(0, times):
        hex = s[i * 8 : i * 8 + 8]
        hex = hex[6:8] + hex[4:6] + hex[2:4] + hex[0:2] # large end -> little end
        bin = ''
        for c in hex:
            bin = bin + hex2bin[c]
        sign = bin[0]
        e = -127
        sum = 128
        for j in range(1, 9):
            num = 0 if bin[j] == '0' else 1
            e = e + num * sum
            sum = sum // 2
        t = 1.0
        sum_f = 0.5
        for j in range(9, 32):
            num = 0 if bin[j] == '0' else 1
            t = t + num * sum_f
            sum_f = sum_f / 2
        fRes = t * (2 ** e)
        if(sign == '1'):
            fRes = - fRes
        if(abs(fRes)< 10e-38): # approach to 0 then 0
            fRes = 0.0
        float_list.append(fRes)

    return float_list

# token2vec
csv_reader = csv.reader(open(csv_file2, encoding='utf-8'))
next(csv_reader) # jump headLine
for row in csv_reader:
    models_vec = []
    model_vec = []
    row[0] = row[0].replace('[','') 
    row[0] = row[0].replace(']','')
    row[0] = row[0].replace("'",'')
    row[0] = row[0].replace(' ','') # delete space
    row[0] = row[0].split(',')
    op_vec = []
    op = []
    # ob_model: one op, one vector
    for token in row[0]:
        if(token == 'EOOperation'):
            op.append(token)
            # processing current Op
            if op[0] == 'return':  # 'return' Op
                op_vec.append(INT_MAX)
                op_id = ob_model_token2vec[op[0]]
                topo = [int(op[1])]
                while len(topo) < d_model_topo_size:
                    topo.append(topo_pad)
                shape=[]
                j = 3 # return Op:  return %xxx torch.vtensor shape_info EOOperation
                while op[j] != 'EOOperation':
                    shape.append(int(op[j]))
                    j = j + 1
                while len(shape) < d_model_shape_size:
                    shape.insert(0, shape_pad)
                for num in op_id:
                    op_vec.append(num)
                for num in topo:
                    op_vec.append(num)
                for num in parameters:
                    op_vec.append(num)
                for num in shape:
                    op_vec.append(num)
                
            else:
                SN = int(op[0])
                op_vec.append(SN)
                if op[1] == 'torch.vtensor': # start Op
                    op_id = [0 for _ in range(d_model_OpId_size)]
                    topo = [topo_pad for _ in range(d_model_topo_size)]
                    parameters = [parameters_pad for _ in range(d_model_params_size)]
                    shape = []
                    j = 2
                    while op[j] != 'EOOperation':
                        shape.append(int(op[j]))
                        j = j + 1
                    while len(shape) < d_model_shape_size:
                        shape.insert(0, shape_pad)
                    for num in op_id:
                        op_vec.append(num)
                    for num in topo:
                        op_vec.append(num)
                    for num in parameters:
                        op_vec.append(num)
                    for num in shape:
                        op_vec.append(num)
                else:

                    # 4 different Ops
                    #   SN      Op_Id                   topo        parameters    shape          
                    #   √       torch.vtensor.literal                   √           √
                    #   √       torch.constant                          √
                    #   √       torch.prim.ListConstruct  √(optional)                     
                    #   √       others                    √                         √

                    if op[1] == 'torch.vtensor.literal':
                        op_id = ob_model_token2vec[op[1]]
                        topo = [topo_pad for _ in range(d_model_topo_size)]
                        parameters = []
                        shape = []
                        j = 3 # point to 'torch.vtensor' while '0x...'
                        # SN, torch.vtensor.literal, '0x...', torch.vtensor, shape_info, EOOperation
                        if op[2].find('0x') == -1:
                            j = 2
                            while op[j] != 'torch.vtensor': # final point to 'torch.vtensor'
                                parameters.append(float(op[j]))
                                j = j + 1
                        else:
                            parameters = hex2float(op[2])

                        # get shape then check： Are all elements same?
                        j = j + 1 # jump 'torch.vtensor'
                        while op[j] != 'EOOperation':
                            shape.append(int(op[j]))
                            j = j + 1
                        total_elements = 1
                        for num in shape:
                            total_elements = total_elements * num
                        if len(parameters) == 1:
                            repeat_elements = parameters[0]
                            while len(parameters) < total_elements:
                                parameters.append(repeat_elements)
                        while len(shape) < d_model_shape_size:
                            shape.insert(0, shape_pad)
                        # processing parameters
                        opNum = 1 # copy times = opNum - 1
                        if len(parameters) <= d_model_params_size:
                            while len(parameters) < d_model_params_size:
                                parameters.append(parameters_pad)
                        else:
                            len_param = len(parameters)
                            opNum = len_param // d_model_params_size
                            remain = len_param % d_model_params_size
                            parameters_align = parameters[0: len_param - remain]
                            if remain > 0:
                                remain_list =  parameters[len_param - remain: len_param]
                                for _ in range(remain, d_model_params_size):
                                    remain_list.append(parameters_pad)
                                for num in remain_list:
                                    parameters.append(num)
                                opNum = opNum + 1
                        while len(shape) < d_model_shape_size:
                            shape.insert(0, shape_pad)
                        for num in op_id:
                            op_vec.append(num)
                        for num in topo:
                            op_vec.append(num)
                        for count in range(0, d_model_params_size):
                            op_vec.append(0) # 0: placeHolder
                        for num in shape:
                            op_vec.append(num)
                        # it is different from other Ops
                        parameters_begin = d_model_SN_size + d_model_OpId_size + d_model_topo_size
                        parameters_end = parameters_begin + d_model_params_size
                        for i in range(0, opNum - 1): # -1: keep same with other Ops
                            op_vec[parameters_begin:parameters_end] = parameters[i*d_model_params_size: i*d_model_params_size+d_model_params_size]
                            model_vec.append(op_vec)
                        # i == opNum
                        op_vec[parameters_begin:parameters_end] = parameters[len(parameters) - d_model_params_size:len(parameters)]
                    elif len(op[1]) >= 14 and op[1][0:14] == 'torch.constant':
                        op_id = ob_model_token2vec[op[1]]
                        topo = [topo_pad for _ in range(d_model_topo_size)]
                        parameters = [parameters_pad for _ in range(d_model_params_size)]
                        parameters[0] = int(op[2]) # only one parameter
                        shape = [1, 1, 1, 1]
                        for num in op_id:
                            op_vec.append(num)
                        for num in topo:
                            op_vec.append(num)
                        for num in parameters:
                            op_vec.append(num)
                        for num in shape:
                            op_vec.append(num)
                    elif op[1] == 'torch.prim.ListConstruct':
                        op_id = ob_model_token2vec[op[1]]
                        topo = []
                        if op[2][0]>='0' and op[2][0] <='9':
                            j = 2
                            while op[j] != 'torch.vtensor' and op[j] != 'EOOperation':
                                topo.append(int(op[j]))
                                j = j + 1
                        while len(topo) < d_model_topo_size:
                            topo.append(topo_pad)
                        parameters = [parameters_pad for _ in range(d_model_params_size)]
                        shape = [0, 0, 0, 0]
                        for num in op_id:
                            op_vec.append(num)
                        for num in topo:
                            op_vec.append(num)
                        for num in parameters:
                            op_vec.append(num)
                        for num in shape:
                            op_vec.append(num)
                    else: # other Ops
                        op_id = ob_model_token2vec[op[1]]
                        topo = []
                        j = 2
                        while op[j] != 'torch.vtensor':
                            topo.append(int(op[j]))
                            j = j + 1
                        while len(topo) < d_model_topo_size:
                            topo.append(topo_pad)
                        parameters = [parameters_pad for _ in range(d_model_params_size)]
                        j = j + 1 # jump 'torch.vtensor'
                        shape = []
                        while op[j] != 'EOOperation':
                            shape.append(int(op[j]))
                            j = j + 1
                        while len(shape) < d_model_shape_size:
                            shape.insert(0, shape_pad)
                        for num in op_id:
                            op_vec.append(num)
                        for num in topo:
                            op_vec.append(num)
                        for num in parameters:
                            op_vec.append(num)
                        for num in shape:
                            op_vec.append(num)   
            op.clear()
            if len(op_vec) != d_model:
                print('opvec_size != d_model, run error!')
                exit()
            model_vec.append(copy.deepcopy(op_vec))
            op_vec.clear()
        else:
            op.append(token)
    models_vec.append(copy.deepcopy(model_vec))
    model_vec.clear()

    # origin_model: one word, one vector
    row[1] = row[1].replace('[','') 
    row[1] = row[1].replace(']','')
    row[1] = row[1].replace("'",'')
    row[1] = row[1].replace(' ','') # delete space
    row[1] = row[1].split(',')
    for token in row[1]:
        model_vec.append(origin_model_token2vec[token])
    models_vec.append(copy.deepcopy(model_vec))
    with open(csv_file3, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(models_vec)
    csv_file.close()
        