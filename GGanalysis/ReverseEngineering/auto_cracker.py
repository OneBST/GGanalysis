'''
用于自动根据综合概率构建抽卡模型
'''
import numpy as np
from GGanalysis.distribution_1d import p2dist, calc_expectation

class LinearAutoCracker():
    '''**概率线性上升软保底模型自动解析工具**

    - ``base_p`` : 基础概率
    - ``avg_p`` ：综合概率
    - ``hard_pity`` ：硬保底抽数
    - ``pity_begin`` ：概率开始上升位置（可选）
    - ``early_hard_pity`` ：是否允许实际硬保底位置提前，默认为否
    - ``forced_hard_pity`` ：是否允许概率上升曲线中硬保底位置概率小于1（此时硬保底位置会被直接设为1）
    
    '''
    def __init__(self, base_p, avg_p, hard_pity, pity_begin=None, early_hard_pity=False, forced_hard_pity=False) -> None:
        self.base_p = base_p
        self.avg_p = avg_p
        self.hard_pity = hard_pity
        self.pity_begin = pity_begin
        self.early_hard_pity = early_hard_pity
        self.forced_hard_pity = forced_hard_pity
        pass
    
    def calc_avg_p(self, p_list):
        '''根据概率提升表计算综合概率'''
        return 1/calc_expectation(p2dist(p_list))
    
    def search_params(self, step_value=0.001, min_pity_begin=1):
        '''搜索符合要求的概率上升表，返回比设定综合概率高或低的参数组合中最接近的
        
        - ``step_value`` : 概率上升值变动的最小值
        - ``min_pity_begin`` : 设定最早的概率开始上升位置
        '''
        p_list = np.zeros(self.hard_pity+1)
        upper_pity_pos = -1
        upper_step_p = -1
        upper_p = -1
        lower_pity_pos = -1
        lower_step_p = -1
        lower_p = -1

        pity_begin_list = range(min_pity_begin, self.hard_pity+1)
        if self.pity_begin is not None:
            pity_begin_list = [self.pity_begin]
        for i in pity_begin_list:
            step_p = 0
            while(step_p < 1):
                step_p += step_value
                p_list[:i] = self.base_p
                p_list[i:] = np.arange(1, self.hard_pity-i+2) * step_p + self.base_p
                if p_list[self.hard_pity] < 1 and not self.forced_hard_pity:
                    # 是否不满足最末尾上升到1
                    continue
                if p_list[self.hard_pity-1] > 1 and not self.early_hard_pity:
                    # 是否在硬保底位置前上升到1
                    break
                p_list = np.minimum(1, p_list)
                p_list[-1] = 1
                estimated_p = self.calc_avg_p(p_list)
                if estimated_p >= self.avg_p:
                    if estimated_p - self.avg_p < abs(upper_p - self.avg_p):
                        upper_pity_pos = i
                        upper_step_p = step_p
                        upper_p = estimated_p
                else:
                    if self.avg_p - estimated_p < abs(self.avg_p - lower_p):
                        lower_pity_pos = i
                        lower_step_p = step_p
                        lower_p = estimated_p
        return (self.base_p, upper_pity_pos, upper_step_p, upper_p), (self.base_p, lower_pity_pos, lower_step_p, lower_p)

if __name__ == '__main__':
    genshin_cracker = LinearAutoCracker(0.006, 0.016, 90)
    print(genshin_cracker.search_params(step_value=0.01))
    hsr_weapon_cracker = LinearAutoCracker(0.008, 0.0187, 80)
    print(hsr_weapon_cracker.search_params(step_value=0.01))
    wuwa_cracker = LinearAutoCracker(0.008, 0.018, 80, pity_begin=66, forced_hard_pity=True)
    print(wuwa_cracker.search_params(step_value=0.04))