import copy


def getSNdict(l: list) -> dict:  # torch-mlir: num% -> OpNum
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


def str2token(s: str) -> list:  # torch-mlir string-> torch-mlir words
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
def get_simple_model_token(
    tokens: list,
) -> list:  # torch-mlir words -> words(delete parameters)
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


# distance
# torch.dict(tensor1, tensor2)


# torch-mlir to model sequences
def get_sequences(s: str) -> list:

    return get_simple_model_token(str2token(s))


def get_SER(tgt_idx, src_idx) -> float:
    """
    get ER:
        ER = SED(tgt_idx, src_idx) / len(src_idx), SED is Sequence Edit Distance.

        params:
            tgt_idx: 目标序列, 不仅仅是算子, 还包括拓扑、形状信息等
            src_idx: 同上
    """

    m = len(tgt_idx)
    n = len(src_idx)

    # 方便理解
    m += 1
    n += 1

    temp = [0 for _ in range(0, n)]
    dp = [copy.deepcopy(temp) for _ in range(0, m)]  # deepcopy

    # init
    for i in range(0, m):
        dp[i][0] = i  # delete: i times
    for i in range(0, n):
        dp[0][i] = i

    # begin
    for i in range(1, m):  # start from 1: compare index 0
        for j in range(1, n):
            if tgt_idx[i - 1] == src_idx[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # no increase
            else:
                tmp = min(dp[i - 1][j], dp[i][j - 1])
                dp[i][j] = min(tmp, dp[i - 1][j - 1]) + 1

    return dp[m - 1][n - 1] / len(src_idx)
