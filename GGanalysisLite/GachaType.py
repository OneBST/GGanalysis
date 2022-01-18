import numpy as np
from scipy import signal
from GGanalysisLite.ConvDist import *

# 保底概率类（可计算截尾分布）
class PityGacha():
    def __init__(self, pity_p=None):
        # 没有传入保底概率表则使用默认概率表
        if pity_p is None:
            self.pity_p = self.generate_pity_p()
        else:
            self.pity_p = pity_p
        self.distribution = self.calc_distribution(self.pity_p)
    # 设置保底概率表
    def generate_pity_p(self):
        pity_p = np.zeros(91, dtype=float)
        for i in range(1, 74):
            pity_p[i] = 0.006
        for i in range(74, 91):
            pity_p[i] = 0.06 + pity_p[i-1]
        pity_p[90] = 1
        return pity_p
    # 根据保底概率表计算截断分布
    def calc_distribution(self, pity_p=[0, 1]):
        item_distribution = np.zeros(len(pity_p), dtype=float)
        temp_state = 1
        for i in range(1, len(pity_p)):
            item_distribution[i] = temp_state * pity_p[i]
            temp_state = temp_state * (1-pity_p[i])
        return item_distribution
    # 以列表形式返回抽取1-n个物品的分布列，比单独计算更快
    def calc_distribution_1_to_n(self, n):
        return smart_conv_1_to_n(self.distribution, n, method='auto')
    # 计算条件下的分布列
    def conditional_distribution(self, n, pull_state=0):  # n为要抽多少个，pull_state为垫抽数量
        # 计算抽一个的条件分布
        temp = self.distribution.copy()
        temp = temp[pull_state:]
        temp[0] = 0
        temp = temp/temp.sum()
        # 计算总分布
        return signal.convolve(temp, smart_conv(self.distribution, n-1))
    # 根据分布计算期望
    def calc_expectation(self, item_distribution=None):
        if item_distribution is None:
            item_distribution = self.distribution
        item_expectation = 0
        for i in range(1, len(item_distribution)):
            item_expectation += item_distribution[i] * i
        return item_expectation

class Pity5starCommon(PityGacha):
    pass

class Pity5starCharacter(PityGacha):
    pass

# 原神UP机制类
class Up5starCharacter(Pity5starCharacter):
    def __init__(self, pity_p=None):
        super().__init__(pity_p)
        self.distribution_5star = self.distribution  # 为预测保留的模块
        temp_distribution = signal.convolve(self.distribution, self.distribution)
        self.distribution = np.pad(self.distribution, (0, len(self.pity_p)-1), 'constant')
        self.distribution = (self.distribution + temp_distribution)/2
    def conditional_distribution(self, n, pull_state=0, up_guarantee=0):
        # n为要抽多少个，pull_state为垫抽数量，up_guarantee为是否有大保底

        # 计算抽一个的条件分布
        if up_guarantee:  # 有大保底
            temp = self.distribution_5star.copy()
            temp = temp[pull_state:]
            temp[0] = 0
            temp = temp/temp.sum()
        else:  # 没有大保底
            temp = self.distribution_5star.copy()
            temp = temp[pull_state:]
            temp[0] = 0
            temp = temp/temp.sum()
            temp2 = signal.convolve(temp, self.distribution_5star)
            temp = np.pad(temp, (0, len(temp2)-len(temp)), 'constant')
            temp = (temp + temp2)/2
        
        # 计算总分布
        if n == 1:
            return temp
        return signal.convolve(temp, smart_conv(self.distribution, n-1))
# 原神武器池基础类
class Pity5starWeapon(PityGacha):
    def generate_pity_p(self):
        pity_p = np.zeros(81, dtype=float)
        for i in range(1, 63):
            pity_p[i] = 0.007
        for i in range(63, 74):
            pity_p[i] = pity_p[i-1] + 0.07
        for i in range(74, 80):
            pity_p[i] = pity_p[i-1] + 0.035
        pity_p[80] = 1
        return pity_p
# 原神「神铸定轨」武器池类（还没写，GGanalysis项目里可以算，没迁移过来）

# 原神玩家类
class GenshinPlayer():
    def __init__(self,
                p5=0,       c5=0,       u5=0,       w5=0,       # 五星数量
                p_pull=0,   c_pull=0,   u_pull=0,   w_pull=0,   # 抽到最后一个五星时恰好花费的抽数
                p_left=0,   c_left=0,               w_left=0,   # 最后一个五星后垫的抽数
                                        u_state=0,              # 处于的保底状态，暂时只有角色UP池1表示大保底
                ):
        self.p5 = p5            # 常驻祈愿五星数量
        self.c5 = c5            # 角色祈愿五星数量
        self.u5 = u5            # 角色祈愿UP五星数量
        self.w5 = w5            # 武器祈愿五星数量

        # 抽到最后一个五星时恰好花费的抽数
        self.p_pull = p_pull
        self.c_pull = c_pull
        self.u_pull = u_pull
        self.w_pull = w_pull

        # 最后一个五星后垫抽抽数
        self.p_left = p_left
        self.c_left = c_left
        self.w_left = w_left

        # 角色UP池保底状态
        self.u_state = u_state
    # 计算分布
    def get_p5_dist(self):
        if self.p_pull == 0:
            return np.ones(1, dtype=float)
        temp = Pity5starCommon()
        return smart_conv(temp.distribution, self.p5)
    def get_c5_dist(self):
        if self.c_pull == 0:
            return np.ones(1, dtype=float)
        temp = Pity5starCharacter()
        return smart_conv(temp.distribution, self.c5)
    def get_u5_dist(self):
        if self.u_pull == 0:
            return np.ones(1, dtype=float)
        temp = Up5starCharacter()
        return smart_conv(temp.distribution, self.u5)
    def get_w5_dist(self):
        if self.w_pull == 0:
            return np.ones(1, dtype=float)
        temp = Pity5starWeapon()
        return smart_conv(temp.distribution, self.w5)
    # 计算条件分布
    def conditional_p5_dist(self, n):
        temp = Pity5starCommon()
        return temp.conditional_distribution(n, self.p_left)
    def conditional_c5_dist(self, n):
        temp = Pity5starCharacter()
        return temp.conditional_distribution(n, self.c_left)
    def conditional_u5_dist(self, n):
        temp = Up5starCharacter()
        return temp.conditional_distribution(n, self.c_left, self.u_state)
    def conditional_w5_dist(self, n):
        temp = Pity5starWeapon()
        return temp.conditional_distribution(n, self.w_left)

    # 获得排名 返回两个值，前为小于等于指定抽数之和，大于等于指定抽数之和
    def get_p5_rank(self):
        return self.get_p5_dist()[:(self.p_pull+1)].sum(), self.get_p5_dist()[(self.p_pull):].sum()
    def get_c5_rank(self):
        return self.get_c5_dist()[:(self.c_pull+1)].sum(), self.get_c5_dist()[(self.c_pull):].sum()
    def get_u5_rank(self):
        return self.get_u5_dist()[:(self.u_pull+1)].sum(), self.get_u5_dist()[(self.u_pull):].sum()
    def get_w5_rank(self):
        return self.get_w5_dist()[:(self.w_pull+1)].sum(), self.get_w5_dist()[(self.w_pull):].sum()
    # 综合排名（角色池根据五星数量计算）
    def get_comprehensive_rank(self):
        p5_dist = self.get_p5_dist()
        c5_dist = self.get_c5_dist()
        w5_dist = self.get_w5_dist()
        temp = signal.convolve(p5_dist, c5_dist)
        temp = signal.convolve(temp, w5_dist)
        return temp[:(self.p_pull+self.c_pull+self.w_pull+1)].sum(), temp[(self.p_pull+self.c_pull+self.w_pull):].sum()
    # 综合排名（角色池根据UP五星数量计算）
    def get_comprehensive_rank_Up5Character(self):
        p5_dist = self.get_p5_dist()
        u5_dist = self.get_u5_dist()
        w5_dist = self.get_w5_dist()
        temp = signal.convolve(p5_dist, u5_dist)
        temp = signal.convolve(temp, w5_dist)
        return temp[:(self.p_pull+self.u_pull+self.w_pull+1)].sum(), temp[(self.p_pull+self.u_pull+self.w_pull):].sum()

if __name__ == '__main__':
    pass