import numpy as np
from scipy.special import comb
from GGanalysis.distribution_1d import *
from GGanalysis.basic_models import PityModel

def calc_stationary_distribution(M):
    '''
    计算转移矩阵对应平稳分布

    转移矩阵如下，平稳分布为列向量
    |1 0.5|   |x|
    |0 0.5|   |y|
    所得平稳分布为转移矩阵特征值1对应的特征向量
    '''
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

def multi_item_rarity(pity_p: list, once_pull_times: int, is_complete=False):
    '''
    计算连抽情况下获得多个道具的概率
    仅仅适用于保底抽卡模型

    注意，当前仅适用于保底开始位置 > 连抽数量的情况
    TODO 更加泛化模型，减少保底开始位置 > 连抽数量的使用限制
    
    采用三阶段计算，主要是因为直接计算数值收敛性不好，此方法兼顾速度和准确
    先得出一直连抽后的保底情况平稳分布，再在平稳分布的基础上计算连抽时首个道具的分布位置
    最后根据第一个分布位置，使用组合数算出概率
    '''
    def build_n_time_matrix(pity_p, item_model, once_pull_times):
        '''
        构建连抽后保底情况的平稳分布
        '''
        M = np.zeros((len(pity_p)-1, len(pity_p)-1), dtype=np.double)
        base_p = pity_p[1]
        for i in range(len(pity_p)-1):
            # 枚举垫抽数量
            # 获得抽到第一个物品的分布
            dist = item_model(1, pull_state=i).dist[:once_pull_times+1]
            dist = pad_zero(dist, once_pull_times+1)
            # 假设概率提升段大于每次连抽次数，这个情况下只需要考虑基础概率
            for j in range(1, once_pull_times+1):
                # 枚举在第 j 抽抽到第一个道具的情况
                p_j = dist[j]
                # 此后都没有抽到的情况，设置转移概率
                M[once_pull_times-j][i] += p_j * (1-base_p) ** (once_pull_times-j)
                for k in range(j+1, once_pull_times+1):
                    # 枚举最后一抽在第k抽处的情况
                    M[once_pull_times-k][i] += p_j * base_p * (1-base_p) ** (once_pull_times-k) 
            # 处理没有抽到的情况
            if once_pull_times+i < len(pity_p)-1:
                M[once_pull_times+i][i] = 1 - sum(dist)
        return M

    def build_n_time_matrix_complete(pity_p, item_model, once_pull_times):
        '''
        考虑概率提升段小大于每次连抽次数的情况
        构建连抽后保底情况的平稳分布

        TODO 验证这个函数的正确性
        '''
        # 计算抽 k 抽都没有道具的概率
        P_m = np.zeros(once_pull_times+1, dtype=float)
        P_m[0] = 1
        for i in range(1, once_pull_times+1):
            if i < len(pity_p):
                P_m[i] = P_m[i-1] * (1-pity_p[i])
            else:
                P_m[i] = 0

        M = np.zeros((len(pity_p)-1, len(pity_p)-1), dtype=np.double)
        base_p = pity_p[1]
        for i in range(len(pity_p)-1):
            # 枚举垫抽数量
            # 获得抽到 1-once_pull_times 个物品的分布
            dists = item_model(once_pull_times, pull_state=i, multi_dist=True)
            for j in range(len(dists)):
                dists[j] = pad_zero(dists[j].dist[:once_pull_times+1], once_pull_times+1)
            for j in range(1, once_pull_times+1):
                # 枚举抽到 j 个道具的情况
                for k in range(j, once_pull_times+1):
                    # 枚举在第 k 抽抽到第 j 个道具且之后不再抽到的情况
                    p_k = dists[j][k] * P_m[once_pull_times-k]
                    M[once_pull_times-k][i] += p_k
            # 处理没有抽到的情况
            if once_pull_times+i < len(pity_p)-1:
                M[once_pull_times+i][i] = 1 - sum(dists[1])
        return M
    
    item_model = PityModel(pity_p)

    # 计算连抽后剩余保底
    if is_complete:
        M = build_n_time_matrix_complete(pity_p, item_model, once_pull_times)
    else:
        M = build_n_time_matrix(pity_p, item_model, once_pull_times)
    stationary_left = calc_stationary_distribution(M)

    # 计算平稳情况下第一个道具位置的分布
    first_item_p = np.zeros(once_pull_times+1, dtype=np.double)
    for i in range(len(pity_p)-1):
        dist = pad_zero(item_model(1, pull_state=i).dist[:once_pull_times+1], once_pull_times+1)
        first_item_p += stationary_left[i] * dist
    
    # 计算连抽出多个道具的概率
    ans = np.zeros(once_pull_times+1, dtype=np.double)
    for i in range(1, once_pull_times+1):
        # 枚举每次连抽获得多少个道具
        for j in range(1, once_pull_times+2-i):
            # 枚举第一个道具位置
            ans[i] += first_item_p[j] * comb(once_pull_times-j, i-1) * pity_p[1] ** (i-1) * (1-pity_p[1]) ** (once_pull_times-j-i+1)
    ans[0] = 1 - sum(ans[1:])
    return ans


class PriorityPitySystem(object):
    """
    不同道具按照优先级排序的保底系统
    若道具为固定概率p，则传入列表填为 [0, p]
    """
    def __init__(self, item_p_list: list, extra_state = 1, remove_pity = False) -> None:
        # TODO extra_state 设置为0会产生问题，需要纠正
        self.item_p_list = item_p_list  # 保底参数列表 按高优先级到低优先级排序
        self.item_types = len(item_p_list)  # 一共有多少种道具
        self.remove_pity = remove_pity
        self.extra_state = extra_state  # 对低优先级保底道具的情况额外延长几个状态，默认为1
        
        self.pity_state_list = []  # 记录每种道具保留几个状态
        self.pity_pos_max = []  # 记录每种道具没有干扰时原始保底位置
        for pity_p in item_p_list:
            self.pity_state_list.append(len(pity_p)+extra_state-1)
            self.pity_pos_max.append(len(pity_p)-1)

        self.max_state = 1  # 最多有多少种状态
        for pity_state in self.pity_state_list:
            self.max_state = self.max_state * pity_state

        # 计算转移矩阵并获得平稳分布
        self.transfer_matrix = self.get_transfer_matrix()  
        self.stationary_distribution = calc_stationary_distribution(self.transfer_matrix)

    def item_pity_p(self, item_type, p_pos):
        # 获得不考虑相互影响情况下的保底概率
        return self.item_p_list[item_type][min(p_pos, self.pity_pos_max[item_type])]

    def get_state(self, state_num) -> list:
        """
        根据状态编号获得保底情况
        """
        pity_state = []
        for i in self.pity_state_list[::-1]:
            pity_state.append(state_num % i)
            state_num = state_num // i
        return pity_state[::-1]

    def get_number(self, pity_state) -> int:
        """
        根据保底情况获得状态编号
        """
        number = 0
        last = 1
        for i, s in zip(self.pity_state_list[::-1], pity_state[::-1]):
            number += s * last
            last *= i
        return number

    def get_next_state(self, pity_state, get_item=None) -> list:
        """
        返回下一个状态
        
        pity_state 为当前状态 get_item 为当前获取的道具，若为 None 则表示没有获得数据
        """
        next_state = []
        for i in range(self.item_types):
            if get_item == i:  # 获得了此类物品
                next_state.append(0)
            else:  # 没有获得此类物品
                # 若有高优先级清除低优先级保底的情况
                if self.remove_pity and get_item is not None:
                    # 本次为低优先级物品
                    if i > get_item:
                        next_state.append(0)
                        continue
                # 没有高优先级清除低优先级保底的情况
                next_state.append(min(self.pity_state_list[i]-1, pity_state[i]+1))
        return next_state

    def get_transfer_matrix(self) -> np.ndarray:
        """
        根据当前的设置生成转移矩阵
        """
        M = np.zeros((self.max_state, self.max_state))  # 状态编号从0开始，0也是其中一种状态

        for i in range(self.max_state):
            left_p = 1
            current_state = self.get_state(i)
            # 本次获得了道具
            for item_type, p_pos in zip(range(self.item_types), current_state):
                next_state = self.get_next_state(current_state, item_type)
                transfer_p = min(left_p, self.item_pity_p(item_type, p_pos+1))
                M[self.get_number(next_state)][i] = transfer_p
                left_p = left_p - transfer_p
            # 本次没有获得任何道具
            next_state = self.get_next_state(current_state, None)
            M[self.get_number(next_state)][i] = left_p
        return  M
    
    def get_stationary_p(self) -> list:
        """
        以列表形式返回每种道具的综合概率
        """
        stationary_p = np.zeros(len(self.item_p_list))
        for i in range(self.max_state):
            current_state = self.get_state(i)
            # 将当前状态的概率累加到对应物品上
            for j, item_state in enumerate(current_state):
                if item_state == 0:
                    stationary_p[j] += self.stationary_distribution[i]
                    # 高优先级优先，不计入低优先级物品
                    break
        return stationary_p

    def get_type_distribution(self, type) -> np.ndarray:
        """
        获取对于某一类道具花费抽数的分布（平稳分布的情况）
        """
        ans = np.zeros(self.pity_state_list[type]+1)
        # print('shape', ans.shape)
        for i in range(self.max_state):
            left_p = 1
            current_state = self.get_state(i)
            for item_type, p_pos in zip(range(type), current_state[:type]):
                left_p -= self.item_pity_p(item_type, p_pos+1)
            # 转移概率
            transfer_p = min(max(0, left_p), self.item_pity_p(type, current_state[type]+1))
            next_pos = min(self.pity_state_list[type], current_state[type]+1)
            ans[next_pos] += self.stationary_distribution[i] * transfer_p
        return ans/sum(ans)