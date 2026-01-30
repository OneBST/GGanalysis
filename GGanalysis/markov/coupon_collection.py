import numpy as np
from typing import Union
from functools import lru_cache
import random

class GeneralCouponCollection():
    '''
        不同奖券概率不均等的奖券集齐问题类
        默认以集齐为吸收态，状态按照从低到高位对应输入列表的从前往后
    '''
    def __init__(self, p_list: Union[list, np.ndarray], item_names: list[str]=None) -> None:
        # 输入合法性检查
        # TODO 考虑是否添加多个吸收态的功能，还是出于速度考虑用DP代替这部分功能
        if len(p_list) >= 16:
            # 默认最高道具种数为 16 种，以防状态过多
            raise ValueError("Item types must below 16")
        sum_p = 0
        for p in p_list:
            if p <= 0:
                raise ValueError("Each p must larger than 0!")
            sum_p += p
        if sum_p > 1:
            raise ValueError("Sum P is greater than 1!")
        if item_names is not None:
            if len(item_names) != len(p_list):
                raise ValueError("item_name must have equal len with p_list!")
        # 类赋值
        self.p_list = np.array(p_list)  # 目标物品的概率，其概率和可以小于1
        self.item_names = item_names  # 道具名称 考虑用字典存其编号
        if item_names is not None:
            # 建立编号查找表，按照列表中的顺序进行编号
            self.item_dict = {}
            for i, item in enumerate(self.item_names):
                self.item_dict[item] = i
        self.fail_p = 1 - sum(p_list)  # 不是目标中的任意一个的概率
        self.item_types = len(p_list)
        self.default_init_state = 0  # 默认起始状态为所有种类都没有
        self.default_target_state = 2 ** self.item_types - 1  # 默认目标状态为集齐状态
    
    def set_default_init_state(self, init_state):
        # 设置默认起始状态
        self.default_init_state = init_state

    def set_default_target_state(self, target_state):
        # 设置默认目标状态
        self.default_target_state = target_state

    def encode_state_number(self, item_list):
        '''根据道具列表生成对应状态'''
        ans = 0
        for item in item_list:
            ans = ans | (1 << self.item_dict[item])
        return ans
    
    def decode_state_number(self, state_num):
        '''根据对应状态返回状态道具列表'''
        ans = []
        for i in range(self.item_types):
            if state_num & (1 << i):
                ans.append(self.item_names[i])
        return ans

    def get_satisfying_state(self, target_state=None):
        '''返回一个标记了满足目标状态要求的01numpy数组'''
        if target_state is None:
            target_state = self.default_target_state
        ans = np.arange(2 ** self.item_types, dtype=int)
        ans = (ans & target_state) == target_state
        return ans.astype(int)

    @lru_cache(maxsize=int(65536))
    def get_expectation(self, state=None, target_state=None):
        '''带缓存递归计算抽取奖券到目标态时抽取数量期望'''
        if state is None:
            state = self.default_init_state
        if target_state is None:
            target_state = self.default_target_state
        if (state & target_state) == target_state:
            # 状态覆盖了目标状态
            return 0
        stay_p = self.fail_p
        temp = 0
        for i, p in enumerate(self.p_list):
            # 枚举本次抽到的奖券
            next_state = state | (1 << i)
            if next_state == state:
                # 这里必须是 += 抽到多个情况都可能保持在原地
                stay_p += p
                continue
            temp += p * self.get_expectation(next_state, target_state)
        return (1+temp) / (1-stay_p)

    def collection_dp(self, n, init_state=None):
        '''通过DP计算抽n次后的状态分布，返回DP数组'''
        if init_state is None:
            init_state = self.default_init_state
        M = np.zeros((self.default_target_state+1, n+1))
        M[init_state, 0] = 1
        for t in range(n):
            for current_state in range(self.default_target_state+1):
                M[current_state, t+1] += M[current_state, t] * self.fail_p
                for i, p in enumerate(self.p_list):
                    next_state = current_state | (1 << i)
                    M[next_state, t+1] += M[current_state, t] * p
        return M

    def get_collection_p(self, n, init_state=None, target_state=None, DP_array=None):
        '''返回抽n抽后达到目标状态的概率数组'''
        if init_state is None:
            init_state = self.default_init_state
        if target_state is None:
            target_state = self.default_target_state
        satisfying_states = self.get_satisfying_state(target_state)
        if DP_array is not None:
            return satisfying_states.dot(DP_array)
        return satisfying_states.dot(self.collection_dp(n, init_state))

    def sim_collection(self, state=None):
        '''蒙特卡洛模拟抽取次数，用于验证'''
        if state is None:
            state = self.default_init_state
        counter = 0
        while((state & self.default_target_state) != self.default_target_state):
            counter += 1
            rand_num = random.random()
            for i, p in enumerate(self.p_list):
                if rand_num < p:
                    state = state | (1 << i)
                    break
                rand_num -= p
        return counter

def get_equal_coupon_collection_exp(item_type, init_type=0, target_type=None):
    '''获得每样道具均等情况下的集齐期望'''
    ans = 0
    if target_type is None: 
        target_type = item_type
    if target_type > item_type:
        raise ValueError("target_type can't be greater than item_types!")
    if init_type < 0:
        raise ValueError("init_type can't below 0!")
    for i in range(init_type, target_type):
        ans += item_type/(item_type-i)
    return ans

if __name__ == '__main__':
    pass