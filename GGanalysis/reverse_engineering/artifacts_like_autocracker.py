'''
用于根据统计数据解析类似原神圣遗物模型的副词条权重
'''
import itertools
import math

def calc_weighted_selection_logp(elements: list, perm: list, removed: list=None, sum_each_perm=True):
    '''**计算不放回抽样概率的对数**

    - ``elements`` : 不放回词条权重
    - ``perm`` ：道具当前选中词条的排列
    - ``removed`` ：被排除的词条（如原神圣遗物中的主词条）
    - ``sum_each_perm`` ：是否将同组合下所有排列概率加和（即不考虑词条获取顺序的情况）
    '''
    def calc_p(perm_labels):
        # 计算选择出指定排列的概率
        p = 1
        weight_all = sum(elements)
        if removed is not None:
            for i in removed:
                weight_all -= elements[i]
        for label in perm_labels:
            p *= elements[label] / weight_all
            weight_all -= elements[label]
        return p
    if not sum_each_perm:
        return math.log(calc_p(perm))  
    # 计算每种排列情况之和
    ans = 0
    perm = sorted(perm)
    permutations = itertools.permutations(perm)
    for perm in permutations:
        ans += calc_p(perm)
    return math.log(ans)

def construct_likelihood_function(elements: list, selected_perms: list[list], perm_numbers: list, removes: list, sum_each_perm=True):
    '''**构建当前情况不放回抽样的似然函数**

    - ``elements`` : 不放回词条权重（优化量）
    - ``selected_perms`` ：道具当前选中词条的排列
    - ``perm_numbers`` ：每种排列出现的次数
    - ``removed`` ：每种排列被排除的词条（如原神圣遗物中的主词条）
    - ``sum_each_perm`` ：是否将同组合下所有排列概率加和（即不考虑词条获取顺序的情况）

    似然函数为 :math:`\log{\pord_{all condition}{P_{condition}^{appear times}}}` ，即对样本中每种出现的主词条-副词条组合记录出现次数，
    将每种情况出现的概率乘方其出现次数，将所有情况的值乘在一起取对数即为本次试验的似然函数。
    '''
    return sum(number * calc_weighted_selection_logp(elements, perm, remove, sum_each_perm) for perm, number, remove in zip(selected_perms, perm_numbers, removes))


if __name__ == '__main__':
    from scipy.optimize import minimize
    import numpy as np
    pass