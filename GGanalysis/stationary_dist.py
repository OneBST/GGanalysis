import numpy as np
from scipy.special import comb

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

def iteration_multi_item_rarity(stationary_p, base_p, iteration_times=1, pull_numbers=10):
    '''
    计算不断进行 pull_numbers 连抽情况下，连抽中出现 n 个道具的概率
    注意，pull_numbers 要小于概率开始上升的抽数，否则这个函数不适用
    计算思想相当简单，对于每次连抽，开头第一个道具遇到的垫抽情况的分布都是相同的，而后续道具获取概率是固定的
    虽然这一想法只是近似，但是完全可以使第一个道具的概率提高到一定值来近似平稳分布后的情况
    这样当保底抽数较高，每次连抽数量较多时误差很低
    按照这一思想一直迭代，直到综合概率相符

    这样做有一定误差，不过我已经想到了一个绝妙的新方法来解决这个问题
    不过现在没空整合进来了，过几天再说
    '''
    # 设置初始数值
    first_p = stationary_p
    state_rate = np.zeros(pull_numbers+1, dtype=np.double)
    # 迭代计算
    for iter in range(iteration_times):
        for i in range(1, pull_numbers):
            state_rate[i] = comb(pull_numbers, i) * first_p * base_p ** (i - 1) * (1 - first_p) * (1 - base_p) ** (pull_numbers - 1 - i)
        state_rate[pull_numbers] = first_p * base_p ** (pull_numbers - 1)
        state_rate[0] = 1 - sum(state_rate[1:])
        now_p = sum(state_rate * np.arange(pull_numbers+1) / pull_numbers)

        first_p = stationary_p / now_p * first_p
    # 迭代不收敛警告
    if abs(now_p/stationary_p-1) / stationary_p > 0.000001:
        print('WARNING! Iteration is not convergent!')

    return state_rate

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