from GGanalysis.scored_item import ScoredItem, ScoredItemSet
from GGanalysis.games.genshin_impact.artifact_data import p_sub_stat, test_weight
from itertools import permutations
from functools import lru_cache
from copy import deepcopy
import numpy as np
'''
    原神圣遗物类
    部分代码修改自 https://github.com/ideless/reliq
'''

# 副词条档位
MINOR_RANKS = [7, 8, 9, 10]
# 权重倍数乘数，必须为整数，越大计算越慢精度越高
RANK_MULTI = 1 

STATS_WEIGHTS = {}
def set_using_weight(new_weight: dict):
    '''更换采用权重时要刷新缓存，注意得分权重必须小于等于1'''
    global STATS_WEIGHTS 
    STATS_WEIGHTS = new_weight
    # print('Refresh weight cache!')
    get_init_state.cache_clear()
    get_state_level_up.cache_clear()

def get_total_weight(weights: dict):
    '''获得同字典中所有键的和'''
    ans = 0
    for key in weights.keys():
        ans += weights[key]
    return ans

def get_combinations_p(stats_p: dict, select_num=4):
    '''获得不同副词条组合的概率'''
    ans = {}
    weight_all = get_total_weight(stats_p)
    for perm in permutations(list(stats_p.keys()), select_num):
        # 枚举并计算该排列的出现概率
        p, s = 1, weight_all
        for m in perm:
            p *= stats_p[m] / s
            s -= stats_p[m]
        # 排序得到键，保证每种组合都是唯一的
        perm_key = tuple(sorted(perm))
        if perm_key in ans:
            ans[perm_key] += p
        else:
            ans[perm_key] = p
    return ans

@lru_cache(maxsize=65536)
def get_init_state(stat_comb, default_weight=0) -> ScoredItem:
    '''这个函数计算初始4词条得分分布，及每个词条的条件期望'''
    score_dist = np.zeros(40*RANK_MULTI+1)
    sub_stat_exp = {}
    for m in stat_comb:
        sub_stat_exp[m] = np.zeros(40*RANK_MULTI+1)
    # 枚举4词条的词条数，共4^4=256种
    for i in range(7, 11):
        s1 = STATS_WEIGHTS.get(stat_comb[0], default_weight) * i * RANK_MULTI
        for j in range(7, 11):
            s2 = s1 + STATS_WEIGHTS.get(stat_comb[1], default_weight) * j * RANK_MULTI
            for k in range(7, 11):
                s3 = s2 + STATS_WEIGHTS.get(stat_comb[2], default_weight) * k * RANK_MULTI
                for l in range(7, 11):
                    # s4 为枚举情况的得分
                    s4 = s3 + STATS_WEIGHTS.get(stat_comb[3], default_weight) * l * RANK_MULTI
                    # 采用比例分配
                    L = int(s4)
                    R = L + 1
                    w_L = R - s4
                    w_R = s4 - L
                    R = min(R, 40 * RANK_MULTI)
                    # 记录数据
                    score_dist[L] += w_L
                    sub_stat_exp[stat_comb[0]][L] += i * w_L
                    sub_stat_exp[stat_comb[1]][L] += j * w_L
                    sub_stat_exp[stat_comb[2]][L] += k * w_L
                    sub_stat_exp[stat_comb[3]][L] += l * w_L
                    score_dist[R] += w_R
                    sub_stat_exp[stat_comb[0]][R] += i * w_R
                    sub_stat_exp[stat_comb[1]][R] += j * w_R
                    sub_stat_exp[stat_comb[2]][R] += k * w_R
                    sub_stat_exp[stat_comb[3]][R] += l * w_R
    # 对于256种情况进行归一化 并移除末尾的0，节省一点后续计算
    for m in stat_comb:
        sub_stat_exp[m] = np.divide(sub_stat_exp[m], score_dist, \
                                    out=np.zeros_like(sub_stat_exp[m]), where=score_dist!=0)
        sub_stat_exp[m] = np.trim_zeros(sub_stat_exp[m], 'b')
    score_dist /= 256
    score_dist = np.trim_zeros(score_dist, 'b')
    return ScoredItem(score_dist, sub_stat_exp)

@lru_cache(maxsize=65536)
def get_state_level_up(stat_comb, default_weight=0) -> ScoredItem:
    '''这个函数计算4词条下升一级的分数分布及每个每个分数下副词条的期望'''
    score_dist = np.zeros(10*RANK_MULTI+1)
    sub_stat_exp = {}
    for m in stat_comb:
        sub_stat_exp[m] = np.zeros(10*RANK_MULTI+1)
    # 枚举升级词条及词条数，共4*4=16种
    for stat in stat_comb:
        for j in range(7, 11):
            score = STATS_WEIGHTS.get(stat, default_weight) * j * RANK_MULTI
            # 采用比例分配
            L = int(score)
            R = L + 1
            w_L = R - score
            w_R = score - L
            R = min(R, 10 * RANK_MULTI)
            # 记录数据
            score_dist[L] += w_L
            sub_stat_exp[stat][L] += j * w_L
            score_dist[R] += w_R
            sub_stat_exp[stat][R] += j * w_R
    # 对于16种情况进行归一化 并移除末尾的0，节省一点后续计算
    for m in stat_comb:
        sub_stat_exp[m] = np.divide(sub_stat_exp[m], score_dist, \
                                    out=np.zeros_like(sub_stat_exp[m]), where=score_dist!=0) 
        sub_stat_exp[m] = np.trim_zeros(sub_stat_exp[m], 'b')
    score_dist /= 16
    score_dist = np.trim_zeros(score_dist, 'b')
    return ScoredItem(score_dist, sub_stat_exp)

class GenshinArtifact(ScoredItem):
    '''原神圣遗物类'''
    def __init__(self, main_stat: str='', sub_stats_p: dict={}, stats_weight: dict={}) -> None:
        # 确定可选副词条
        self.main_stat = main_stat
        self.sub_stats_p = deepcopy(sub_stats_p)
        if self.main_stat in self.sub_stats_p:
            del self.sub_stats_p[self.main_stat]
        # 权重改变时应清除缓存
        self.stats_weight = stats_weight
        if self.stats_weight != STATS_WEIGHTS:
            set_using_weight(self.stats_weight)
        # 计算副词条组合概率
        ans = ScoredItem()
        self.sub_stats_combinations = get_combinations_p(stats_p=self.sub_stats_p, select_num=4)
        # 遍历所有情况累加
        for stat_comb in list(self.sub_stats_combinations.keys()):
            temp_base = get_init_state(stat_comb)
            temp_level_up = get_state_level_up(stat_comb)
            # 初始3词条和初始四词条的情况
            temp_3 = temp_base * temp_level_up * temp_level_up * temp_level_up * temp_level_up
            temp_4 = temp_3 * temp_level_up
            ans += (0.8 * temp_3 + 0.2 * temp_4) * self.sub_stats_combinations[stat_comb]
        self.score_dist = ans.score_dist
        self.sub_stats_exp = ans.sub_stats_exp

class GenshinArtifactSet(ScoredItemSet):
    def __init__(self, item_set=...) -> None:
        super().__init__(item_set)

if __name__ == '__main__':
    pass
    '''
    # 测速代码
    import time
    from my_implementation_ideles_conv_sub_stats import combine_score_items
    s = time.time()
    for i in range(1):
        get_init_state.cache_clear()
        get_state_level_up.cache_clear()
        # test = GenshinArtifact('hp', p_sub_stat, test_weight)
        flower = GenshinArtifact('hp', p_sub_stat, test_weight)
        plume = GenshinArtifact('atk', p_sub_stat, test_weight)
        sands = GenshinArtifact('atkp', p_sub_stat, test_weight)
        goblet = GenshinArtifact('pyroDB', p_sub_stat, test_weight)
        circlet = GenshinArtifact('cr', p_sub_stat, test_weight)
        set = [flower, plume, sands, goblet, circlet]
        temp = flower * plume * sands * goblet * circlet
    t = time.time()
    print(t-s)
    '''
    '''
    # 套装备选4+1绘图
    from matplotlib import pyplot as plt
    test = GenshinArtifact('hp', p_sub_stat, test_weight)
    test = test.repeat(1000, 1/5)
    ans = np.outer(test.score_dist.dist, test.score_dist.dist)
    plt.matshow(ans)
    # print(sum(test.score_dist.dist), test.score_dist.dist[0])
    # plt.plot(test.score_dist.dist, color='C1')
    plt.show()
    '''

    # ans_p = 0
    # test = get_combinations_p(P_MINOR, 4)
    # for key in test.keys():
    #     if 'atkp' in key or 'cr' in key or 'cd' in key:
    #         ans_p += test[key]
    # print(ans_p, 1-ans_p)

