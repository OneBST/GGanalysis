import numpy as np
from scipy import signal
from GGanalysisLite.GachaType import *

# 快速幂+FFT卷积
def smart_conv(dist , item_num, method='auto'):
    ans = np.ones(1)
    temp = dist
    t = int(item_num)
    while True:
        if t % 2:
            ans = signal.convolve(ans, temp, method=method)
        t = int(t/2)
        if t == 0:
            break
        temp = signal.convolve(temp, temp, method=method)
    return ans

def get_5star_dist(item_num):
    temp = Pity5starCommon()
    return smart_conv(temp.distribution, item_num)

class GenshinPlayer():
    def __init__(self, p5=0, c5=0, u5=0, w5=0,
                p_pull=0, c_pull=0, u_pull=0, w_pull=0):
        self.p5 = p5            # 常驻祈愿五星数量
        self.c5 = c5            # 角色祈愿五星数量
        self.u5 = u5            # 角色祈愿UP五星数量
        self.w5 = w5            # 武器祈愿五星数量
        self.p_pull = p_pull    # 常驻祈愿抽到最后一个五星时恰好花费的抽数
        self.c_pull = c_pull    # 角色祈愿抽到最后一个五星时恰好花费的抽数
        self.u_pull = u_pull    # 角色祈愿抽到最后一个UP五星时恰好花费的抽数
        self.w_pull = w_pull    # 武器祈愿抽到最后一个五星时恰好花费的抽数
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
    # 获得排名
    def get_p5_rank(self):
        return self.get_p5_dist()[:(self.p_pull+1)].sum()
    def get_c5_rank(self):
        return self.get_c5_dist()[:(self.c_pull+1)].sum()
    def get_u5_rank(self):
        return self.get_u5_dist()[:(self.u_pull+1)].sum()
    def get_w5_rank(self):
        return self.get_w5_dist()[:(self.w_pull+1)].sum()
    # 综合排名（角色池根据五星数量计算）
    def get_comprehensive_rank(self):
        p5_dist = self.get_p5_dist()
        c5_dist = self.get_c5_dist()
        w5_dist = self.get_w5_dist()
        temp = signal.convolve(p5_dist, c5_dist)
        temp = signal.convolve(temp, w5_dist)
        return temp[:(self.p_pull+self.c_pull+self.w_pull+1)].sum()
    # 综合排名（角色池根据UP五星数量计算）
    def get_comprehensive_rank_Up5Character(self):
        p5_dist = self.get_p5_dist()
        u5_dist = self.get_u5_dist()
        w5_dist = self.get_w5_dist()
        temp = signal.convolve(p5_dist, u5_dist)
        temp = signal.convolve(temp, w5_dist)
        return temp[:(self.p_pull+self.u_pull+self.w_pull+1)].sum()

if __name__ == '__main__':
    pass