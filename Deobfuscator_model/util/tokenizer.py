""""
提取主要信息：

    1. 读取data.csv中的数据

    2. 分词同时过滤掉不需要的词汇

    3.写入token.csv中

"""

import csv

csv_file_path = "/home/pingmu123/torch-mlir/Deobfuscator_model/dataSet/"
csv_file1 = csv_file_path + "data.csv"
csv_file2 = csv_file_path + "token.csv"
csv_file2_origin = csv_file_path + "origin_token.csv"
csv_file3 = csv_file_path + "token2vec.csv"
csv_file_header = ['ob_models_input', 'origin_models_label']


# padding
model_token_padding = '<EOS>'

# begin symbol
origin_model_token_beginning = '<BOS>'

with open(csv_file2, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_file_header)
csv_file.close()

with open(csv_file2_origin, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_file_header)
csv_file.close()

csv_reader = csv.reader(open(csv_file1, encoding='utf-8'))
next(csv_reader) # jump headLine

def getSNdict(l:list)->dict:
    SNdict={}
    SN = 0
    for token in l:
        # print(token)
        if token[0] == '%':
            if token not in SNdict:
                SNdict[token] = SN
                SN = SN + 1
            else:
                pass
        else:
            pass
    return SNdict

def str2token(s:str)->list:
    tokens=[]
    # pre processing

    # 1. special: input %arg0 shape
    pos = s.find(':')
    s1 = s[0:pos]
    s2 = s[pos+1:len(s)]
    s = s1 + ' to' + s2  # ' to' not 'to'
    pos1 = s.find('->')
    pos2 = s.find('EOVector')
    s = s[0:pos2] + s[pos2+len('EOVector'):len(s)]
    pos2 = s.find('EOVector') # second EOVector
    s1 = s[0:pos1]
    s2 = s[pos2: len(s)]
    s = s1 + ' ' + s2

    # 2. final ':'
    pos = s.rfind(':')
    s1 = s[0:pos]
    s2 = s[pos+1:len(s)]
    s = s1 + 'to' + s2
    while s.find(':') != -1:
        begin = s.find(':')
        # NOTES: 
        #   if end1 = s[begin+1:len(s)].find('->')
        #   then end1 from 0 to ...
        end1 = begin + 1 + s[begin+1:len(s)].find('->') # +1: find in the behind of ':'
        end2 = begin + 1 + s[begin+1:len(s)].find(':') # for torch.tensor.literal and ...
        if end1 != begin or end2 != begin :
            if (end1 != begin and end2 != begin and end1 < end2) or (end1 != begin and end2 == begin):
                s1 = s[0:begin]
                s2 = s[end1:len(s)] 
                s = s1 + s2 # delete tokens which in middle of ':' and '->', include ':'
            else: # (end1 != begin and end2 != begin and end2 < end1) or (end1 == begin and end2 != begin):
                s1 = s[0:begin]
                s2 = s[end2+1:len(s)]
                s = s1 + 'to' + s2 # delete tokens which in middle of ':' and ':', include ':'
    s = s.replace('->', ' to')
    s = s.replace('(', ' ')
    s = s.replace(')', ' ')
    s = s.replace('<', ' ')
    s = s.replace('>', ' ')
    s = s.replace('[', ' ')
    s = s.replace(']', ' ')
    s = s.replace('{', ' ')
    s = s.replace('}', ' ')
    s = s.replace('0.000000e+00', '0')
    s = s.replace('1.000000e+00', '1')
    # space * n -> space * 1
    for i in range(0, 5): # n <= 6
        s = s.replace('  ', ' ')

    s = s.split(' ')
    pre_token = ''
    for token in s:
        if len(token)>0:
            if token[0] == '%':
                tokens.append(token)
            elif token[0] == '!':
                if(token=='!torch.vtensor'):
                    if pre_token == 'to':
                        tokens.append('torch.vtensor')
            elif token[0] == 't':
                if token == 'torch.debug_module_name' or token == 'tensor' or token == 'to':
                    pass
                elif token == 'true':
                    tokens.append('1')
                else:
                    tokens.append(token)
            elif (token[0] >= '0' and token[0] <='9') or token[0] == '-':
                if token.find('xf32')== -1: # not include 'xf32'
                    tokens.append(token)
                else:
                    pass
            elif token == 'return':
                tokens.append(token)
            elif(token[0]=='"'):
                token = token.replace('"','')
                if token.find('0x')== 0:
                    tokens.append(token)
                else:
                    pass
            elif token == 'EOVector':
                if pre_token != 'EOVector':
                    tokens.append(token)
            elif token == 'false':
                tokens.append('0')
            else:
                pass
        pre_token = token
    SNdict = getSNdict(tokens)
    for i in range(0, len(tokens)):
        if tokens[i] in SNdict:
            tokens[i] = SNdict[tokens[i]]
    return tokens

# delete parameters
def get_simple_origin_model_token(tokens:list)->list:
    res = []
    i = 0
    while i < len(tokens):
        if tokens[i] == 'torch.vtensor.literal':
            res.append(tokens[i])
            while i<len(tokens) and tokens[i]!= 'torch.vtensor':
                i = i + 1 
        else:
            res.append(tokens[i])
            i = i + 1
    return res



for row in csv_reader:
    ob_model = row[0]
    ob_model_token = str2token(ob_model)
    ob_model_token.append(model_token_padding)
    origin_model = row[1]
    origin_model_token = str2token(origin_model)
    origin_model_token.insert(0, origin_model_token_beginning)
    origin_model_token.append(model_token_padding)
    tempRow = [ob_model_token, origin_model_token]
    with open(csv_file2_origin, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(tempRow)
    csv_file.close()
    simple_origin_model_token = get_simple_origin_model_token(origin_model_token)
    tempRow = [ob_model_token, simple_origin_model_token]
    with open(csv_file2, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(tempRow)
    csv_file.close()
