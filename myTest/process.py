import csv


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


model = 'module attributes {torch.debug_module_name = ""testNet""} { EOOperation  func.func @forward(%arg0: !torch.vtensor<[1 1 28 28] f32>) -> !torch.vtensor<[1 2 26 26] f32> { EOOperation    %false = torch.constant.bool false EOOperation    %0 = torch.vtensor.literal(dense<[0.209938765  -0.148417324]> : tensor<2xf32>) : !torch.vtensor<[2] f32> EOOperation    %1 = torch.vtensor.literal(dense<[[[[0.096218273  -0.201871157  0.321646333]  [0.301793039  0.0947734565  0.105118357]  [0.103841625  0.195770621  -0.0249449015]]]  [[[0.304565877  -0.298244804  -0.328105032]  [0.253264964  0.0772231445  -0.280481935]  [-0.0292607155  0.00742471218  3.352710e-02]]]]> : tensor<2x1x3x3xf32>) : !torch.vtensor<[2 1 3 3] f32> EOOperation    %int0 = torch.constant.int 0 EOOperation    %int1 = torch.constant.int 1 EOOperation    %2 = torch.prim.ListConstruct %int1  %int1 : (!torch.int  !torch.int) -> !torch.list<int> EOOperation    %3 = torch.prim.ListConstruct %int0  %int0 : (!torch.int  !torch.int) -> !torch.list<int> EOOperation    %4 = torch.prim.ListConstruct  : () -> !torch.list<int> EOOperation    %5 = torch.aten.convolution %arg0  %1  %0  %2  %3  %2  %false  %4  %int1 : !torch.vtensor<[1 1 28 28] f32>  !torch.vtensor<[2 1 3 3] f32>  !torch.vtensor<[2] f32>  !torch.list<int>  !torch.list<int>  !torch.list<int>  !torch.bool  !torch.list<int>  !torch.int -> !torch.vtensor<[1 2 26 26] f32> EOOperation    return %5 : !torch.vtensor<[1 2 26 26] f32> EOOperation  } EOOperation} EOOperation'
result = str2token(model)
f = open("result.txt", "w")
f.write(str(result))
f.close()
