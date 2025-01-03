import numpy as np
from GGanalysis import FiniteDist

def calc_EP_weapon_classic(a, b):
    '''
    返回原神5.0版本前2.0版本后同卡池抽A限定五星武器a个和B限定五星武器b个所需五星个数分布

    计算5.0前2.0后命定值为2的情况下，恰好花费k个五星后恰好抽到a个A限定UP五星和b个B限定UP五星的概率，
    抽取时采用最优策略，若a=b，则定轨当前离要求数量较远的一个限定UP五星，
    若a≠b，则同样定轨当前离要求数量较远的限定UP五星，当离要求数量一致时，转化为a=b问题，
    若a=b，至多需要获取3*a个五星；若a≠b，至多需要获取3*max(a,b)个五星。
    '''
    # S 第0维表示获得A的数量 第1维表示获得B的数量 第2维表示获得常驻的数量
    S = np.zeros((60, 60, 60))
    S[(0, 0, 0)] = 1
    N = np.zeros((60, 60, 60))
    # 记录最后到达每个状态概率
    ans_dist = np.zeros((60, 60, 60))
    num_dist = np.zeros(200)
    # 控制状态更新，注意同样的状态可以在不同轮进行反复更新

    E = 0  # 期望值
    # 枚举定轨轮数
    for ep in range(3*max(a, b)+1):  # 此处+1是为了扫尾
        for i in range(2*max(a, b)+2):  # 获得A的数量 +2 是为了先去A后再去B使得A多1
            for j in range(2*max(a, b)+1):  # 获得B的数量
                for k in range(a+b+1):  # 获得常驻的数量
                    # 若已经满足条件，则不继续
                    if i>=a and j >=b:
                        # if i+j+k >= 21 and S[i,j,k] != 0:
                        #     print(i, j, k)
                        ans_dist[i,j,k] += S[i,j,k]
                        num_dist[i+j+k] += S[i,j,k]
                        E += (i + j + k) * S[i,j,k]
                        S[i,j,k] = 0  # 虽然是多余的还是可以写在这hhh
                        continue
                    # 定轨A进行抽取的情况
                    if a-i >= b-j:
                        N[i+1,j,k] += (3/8)*S[i,j,k]
                        N[i+1,j+1,k] += (9/64)*S[i,j,k]
                        N[i+1,j,k+1] += (1/8)*S[i,j,k]
                        N[i+1,j+2,k] += (9/64)*S[i,j,k]
                        N[i+1,j+1,k+1] += (7/32)*S[i,j,k]
                    # 定轨B进行抽取的情况
                    if a-i < b-j:
                        N[i,j+1,k] += (3/8)*S[i,j,k]
                        N[i+1,j+1,k] += (9/64)*S[i,j,k]
                        N[i,j+1,k+1] += (1/8)*S[i,j,k]
                        N[i+2,j+1,k] += (9/64)*S[i,j,k]
                        N[i+1,j+1,k+1] += (7/32)*S[i,j,k]
        # 完成一轮后
        S = N
        N = 0 * N
    if abs(np.sum(ans_dist)-1)> 0.00001:
        print("ERROR: sum of ans is not equal to 1!", np.sum(ans_dist))
        exit()
    return FiniteDist(np.trim_zeros(num_dist, 'b'))


def calc_EP_weapon_simple(a, b):
    '''
    返回原神5.0版本后同卡池抽A限定五星武器a个和B限定五星武器b个所需五星个数分布
    
    计算5.0后命定值为1的情况下，恰好花费k个五星后恰好抽到a个A限定UP五星和b个B限定UP五星的概率，
    抽取时采用简单策略，定轨当前离要求数量较远的一个限定UP五星。
    '''
    # S 第0维表示获得A的数量 第1维表示获得B的数量 第2维表示获得常驻的数量
    S = np.zeros((60, 60, 60))
    S[(0, 0, 0)] = 1
    N = np.zeros((60, 60, 60))
    # 记录最后到达每个状态概率
    ans_dist = np.zeros((60, 60, 60))
    num_dist = np.zeros(200)
    # 控制状态更新，注意同样的状态可以在不同轮进行反复更新

    E = 0  # 期望值
    # 枚举定轨轮数
    for ep in range(2*(a+b)+1):  # 此处+1是为了扫尾
        for i in range(a+b+1):  # 获得A的数量
            for j in range(a+b+1):  # 获得B的数量
                for k in range(a+b+1):  # 获得常驻的数量
                    # 若已经满足条件，则不继续
                    if i>=a and j >=b:
                        ans_dist[i,j,k] += S[i,j,k]
                        num_dist[i+j+k] += S[i,j,k]
                        E += (i + j + k) * S[i,j,k]
                        S[i,j,k] = 0  # 虽然是多余的还是可以写在这hhh
                        continue
                    # 定轨A进行抽取的情况
                    if a-i >= b-j:
                        N[i+1,j,k] += (3/8)*S[i,j,k]
                        N[i+1,j+1,k] += (3/8)*S[i,j,k]
                        N[i+1,j,k+1] += (1/4)*S[i,j,k]
                    # 定轨B进行抽取的情况
                    if a-i < b-j:
                        N[i,j+1,k] += (3/8)*S[i,j,k]
                        N[i+1,j+1,k] += (3/8)*S[i,j,k]
                        N[i,j+1,k+1] += (1/4)*S[i,j,k]
        # 完成一轮后
        S = N
        N = 0 * N
    if abs(np.sum(ans_dist)-1)> 0.00001:
        print("ERROR: sum of ans is not equal to 1!", np.sum(ans_dist))
        exit()
    return FiniteDist(np.trim_zeros(num_dist, 'b'))

if __name__ == '__main__':
    pass