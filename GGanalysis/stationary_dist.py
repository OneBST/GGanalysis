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

class PriorityPitySystem(object):
    """
    不同道具按照优先级排序的保底系统
    """
    def __init__(self, item_p_list: list, extra_state = 1, remove_pity = False) -> None:
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

    def get_state(self, state_num):
        """
        根据状态编号获得保底情况
        """
        pity_state = []
        for i in self.pity_state_list[::-1]:
            pity_state.append(state_num % i)
            state_num = state_num // i
        return pity_state[::-1]

    def get_number(self, pity_state):
        """
        根据保底情况获得状态编号
        """
        number = 0
        last = 1
        for i, s in zip(self.pity_state_list[::-1], pity_state[::-1]):
            number += s * last
            last *= i
        return number

    def get_next_state(self, pity_state, get_item=None):
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
                    if i > get_item:
                        next_state.append(0)
                        continue
                next_state.append(min(self.pity_state_list[i]-1, pity_state[i]+1))
        return next_state

    def get_transfer_matrix(self):
        """
        根据当前的设置生成转移矩阵
        """
        # TODO 修改代码使其能应对 extra_state 不为1，且可能高优先级吞掉低优先级的情况
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
    
    def get_stationary_p(self):
        """
        返回每种道具的综合概率
        """
        stationary_p = np.zeros(len(self.item_p_list))
        for i in range(self.max_state):
            current_state = self.get_state(i)
            # 将当前状态的概率累加到对应物品上
            for j, item_state in enumerate(current_state):
                if item_state == 0:
                    stationary_p[j] += self.stationary_distribution[i]
                    break
        return stationary_p

    def get_type_distribution(self, type):
        """
        获取对于某一类道具花费抽数的分布（平稳分布的情况）
        """
        ans = np.zeros(self.pity_state_list[type]+1)
        print('shape', ans.shape)
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