import numpy as np

# 计算转移矩阵对应平稳分布
'''
    转移矩阵如下，分布为列向量
    |1 0.5|     |x|
    |0 0.5|     |y|
'''
def calc_stationary_distribution(M):
    matrix_shape = np.shape(M)
    if matrix_shape[0] == matrix_shape[1]:
        pass
    else:
        print("平稳分布计算错误:输入应该为方阵")
        return
    # 减去对角阵
    C = M - np.identity(matrix_shape[0])
    # 末行设置为1
    C[matrix_shape[0]-1] = 1
    # 设置向量
    X = np.zeros(matrix_shape[0], dtype=float)
    X[matrix_shape[0]-1] = 1
    # 解线性方程求解
    ans = np.linalg.solve(C, X)
    return ans

def calc_coupling_p(p_a, p_b):
    # a为高优先级 b为低优先级
    # 保底抽数
    pos_a = len(p_a)  # 91
    pos_b = len(p_b)  # 11
    # 状态数为pos_a*pos_b个
    # 状态a_state*pos_b+b_state 表示a_state抽没出a且b_state没出b的状态
    M = np.zeros(((pos_a-1)*pos_b, (pos_a-1)*pos_b), dtype=float)
    # 高优先级没有抽到
    for i in range(1, pos_a-1):  # 本抽高优先级物品状态 1-89
        # 抽到了低优先级
        now_state = i*pos_b
        for j in range(pos_b-1):  # 上抽低优先级物品状态 0-9
            pre_state = (i-1)*pos_b + j
            trans_p = min(1-p_a[i], p_b[j+1])
            M[now_state][pre_state] = trans_p
        # 处理被挤走情况 上抽为10
        pre_state = (i-1)*pos_b + pos_b-1
        M[now_state][pre_state] =  1-p_a[i]

        # 低优先级也没有抽到
        for j in range(pos_b-1):  # 上抽低优先级物品状态 0-9
            now_state = i*pos_b + (j+1)
            pre_state = (i-1)*pos_b + j
            trans_p = max(1-p_a[i]-p_b[j+1], 0)
            M[now_state][pre_state] = trans_p
    # 高优先级的抽到了
    for i in range(pos_a-1):  # 上抽高优先级物品状态 0-89
        for j in range(1, pos_b):  # 上抽低优先级物品状态 0-10
            now_state = j  # 稳态时i/j不可能同时为0
            pre_state = i*pos_b + (j-1)
            trans_p = p_a[i+1]
            M[now_state][pre_state] = trans_p
    # 一直出高优先级物品的情况
    for i in range(pos_a-1):  # 上抽高优先级物品状态 0-88
        now_state = pos_b-1
        pre_state = i*pos_b + pos_b-1
        trans_p = p_a[i+1]
        M[now_state][pre_state] = trans_p
    
    # 转移矩阵设置完成后求解平稳分布
    ans = calc_stationary_distribution(M)
    
    # 低优先度物品分布计算
    ans_b = np.zeros(pos_b+1, dtype=float)
    for i in range(pos_a-1):
        for j in range(0, pos_b-1):
            trans_p = min(1-p_a[i+1], p_b[j+1])
            ans_b[j+1] += ans[i*pos_b + j] * trans_p
        trans_p = min(1-p_a[i+1], 1)
        ans_b[pos_b] += ans[i*pos_b + pos_b-1] * trans_p
    return ans_b/ans_b.sum()
