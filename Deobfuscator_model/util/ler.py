def get_ler(tgt_idx, src_idx)->float:
    """
    get LER:
        LER = ED(tgt_idx, src_idx) / len(src_idx), ED is edit distance.
        
        params: 
            tgt_idx: 目标序列, 不仅仅是算子, 还包括拓扑、形状信息等
            src_idx: 同上
    """

    m = len(tgt_idx)
    n = len(src_idx)

    # 方便理解
    m+=1
    n+=1
    
    temp = [0 for _ in range(0, n)]
    dp = [temp for _ in range(0, m)]

    # init
    for i in range(0, m):
        dp[i][0] = i # delete: i times
    for i in range(0, n):
        dp[0][i] = i

    # begin
    for i in range(1, m): # start from 1: compare index 0
        for j in range(1, n):
            if tgt_idx[i-1] == src_idx[i-1]:
                dp[i][j] = dp[i-1][j-1] # not increase
            else:
                tmp = min(dp[i-1][j], dp[i][j-1])
                dp[i][j] = min(tmp, dp[i-1][j-1])
    return dp[m-1][j-1]