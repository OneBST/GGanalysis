from copy import deepcopy
from functools import lru_cache
from itertools import permutations
from typing import Callable

import numpy as np

from GGanalysis.games.genshin_impact.artifact_data import *
from GGanalysis.ScoredItem.scored_item import ScoredItem, ScoredItemSet

"""
    原神圣遗物类
    部分代码修改自 https://github.com/ideless/reliq
"""
__all__ = [
    "GenshinArtifact",
    "GenshinArtifactSet",
    "ARTIFACT_TYPES",
    "STAT_NAME",
    "W_MAIN_STAT",
    "W_SUB_STAT",
    "DEFAULT_MAIN_STAT",
    "DEFAULT_STAT_SCORE",
    "DEFAULT_STAT_COLOR",
]

# 副词条档位
MINOR_RANKS = [7, 8, 9, 10]
# 权重倍数乘数，必须为整数，越大计算越慢精度越高
RANK_MULTI = 1
# 全局圣遗物副词条权重
STATS_WEIGHTS = {}


def set_using_weight(new_weight: dict):
    """更换采用权重时要刷新缓存，注意得分权重必须小于等于1"""
    global STATS_WEIGHTS
    STATS_WEIGHTS = new_weight
    # print('Refresh weight cache!')
    get_init_state.cache_clear()
    get_state_level_up.cache_clear()

def dict_weight_sum(weights: dict):
    """获得字典中所有值的和"""
    return sum(weights.values())

def get_combinations_p(stats_p: dict, select_num=4):
    """获得拥有4个副词条的五星圣遗物不同副词条组合的概率"""
    ans = {}
    weight_all = dict_weight_sum(stats_p)
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
    """获得拥有4个副词条的五星圣遗物初始得分分布，及得分下每个副词条的条件期望"""
    score_dist = np.zeros(40 * RANK_MULTI + 1)
    sub_stat_exp = {}
    for m in stat_comb:
        sub_stat_exp[m] = np.zeros(40 * RANK_MULTI + 1)
    # 枚举4词条的词条数，共4^4=256种
    for i in range(7, 11):
        s1 = STATS_WEIGHTS.get(stat_comb[0], default_weight) * i * RANK_MULTI
        for j in range(7, 11):
            s2 = s1 + STATS_WEIGHTS.get(stat_comb[1], default_weight) * j * RANK_MULTI
            for k in range(7, 11):
                s3 = (
                    s2
                    + STATS_WEIGHTS.get(stat_comb[2], default_weight) * k * RANK_MULTI
                )
                for l in range(7, 11):
                    # s4 为枚举情况的得分
                    s4 = (
                        s3
                        + STATS_WEIGHTS.get(stat_comb[3], default_weight)
                        * l
                        * RANK_MULTI
                    )
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
        sub_stat_exp[m] = np.divide(
            sub_stat_exp[m],
            score_dist,
            out=np.zeros_like(sub_stat_exp[m]),
            where=score_dist != 0,
        )
        sub_stat_exp[m] = np.trim_zeros(sub_stat_exp[m], "b")
    score_dist /= 256
    score_dist = np.trim_zeros(score_dist, "b")
    return ScoredItem(score_dist, sub_stat_exp)

@lru_cache(maxsize=65536)
def get_state_level_up(stat_comb, default_weight=0) -> ScoredItem:
    """这个函数计算4词条下升1级的分数分布及每个每个分数下副词条的期望"""
    score_dist = np.zeros(10 * RANK_MULTI + 1)
    sub_stat_exp = {}
    for m in stat_comb:
        sub_stat_exp[m] = np.zeros(10 * RANK_MULTI + 1)
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
        sub_stat_exp[m] = np.divide(
            sub_stat_exp[m],
            score_dist,
            out=np.zeros_like(sub_stat_exp[m]),
            where=score_dist != 0,
        )
        sub_stat_exp[m] = np.trim_zeros(sub_stat_exp[m], "b")
    score_dist /= 16
    score_dist = np.trim_zeros(score_dist, "b")
    return ScoredItem(score_dist, sub_stat_exp)

class GenshinArtifact(ScoredItem):
    """原神圣遗物类"""

    def __init__(
        self,
        type: str = "flower",  # 道具类型
        type_p = 1/5,  # 每次获得道具是是类型道具概率
        main_stat: str = None,  # 主词条属性
        sub_stats_select_weight: dict = W_SUB_STAT,  # 副词条抽取权重
        stats_score: dict = DEFAULT_STAT_SCORE,  # 词条评分权重
        p_4sub: float = P_DROP_STATE["domains_drop"],  # 根据圣遗物掉落来源确定4件套掉落率
        sub_stats_filter: Callable[..., bool] = None, # 设定副词条过滤器，若函数判断False直接丢掉被过滤的情况
        forced_combinations = None, # 直接设定初始词条组合
    ) -> None:
        # 计算获得主词条概率
        self.type = type
        if main_stat is not None:
            self.main_stat = main_stat
        else:
            self.main_stat = DEFAULT_MAIN_STAT[self.type]
        drop_p = (
            type_p
            * W_MAIN_STAT[self.type][self.main_stat]
            / dict_weight_sum(W_MAIN_STAT[self.type])
        )
        # 确定可选副词条
        self.sub_stats_weight = deepcopy(sub_stats_select_weight)
        if self.main_stat in self.sub_stats_weight:
            del self.sub_stats_weight[self.main_stat]
        # 确定副词条四件概率
        self.p_4sub = p_4sub
        # 词条权重改变时应清除缓存
        self.stats_score = stats_score
        if self.stats_score != STATS_WEIGHTS:
            set_using_weight(self.stats_score)
        # 计算副词条组合概率
        ans = ScoredItem()
        if forced_combinations is None:
            self.sub_stats_combinations = get_combinations_p(
                stats_p=self.sub_stats_weight, select_num=4
            )
        else:
            self.sub_stats_combinations = forced_combinations
        # 遍历所有情况累加
        for stat_comb in list(self.sub_stats_combinations.keys()):
            if sub_stats_filter is not None:
                if sub_stats_filter(stat_comb) is False:
                    continue
            temp_base = get_init_state(stat_comb)
            temp_level_up = get_state_level_up(stat_comb)
            # 初始3词条和初始四词条的情况
            temp_3 = (
                temp_base
                * temp_level_up
                * temp_level_up
                * temp_level_up
                * temp_level_up
            )
            temp_4 = temp_3 * temp_level_up
            ans += (
                (1 - self.p_4sub) * temp_3 + self.p_4sub * temp_4
            ) * self.sub_stats_combinations[stat_comb]
        super().__init__(ans.score_dist, ans.sub_stats_exp, drop_p, stats_score=self.stats_score)

# 导入所需的最优组合组件
from GGanalysis.ScoredItem.scored_item_tools import (
    get_mix_dist,
    remove_worst_combination,
    select_best_combination,
)

class GenshinArtifactSet(ScoredItemSet):
    def __init__(
        self,
        main_stat: dict = DEFAULT_MAIN_STAT,
        stats_score: dict = DEFAULT_STAT_SCORE,
        p_4sub: float = P_DROP_STATE["domains_drop"],  # 根据圣遗物掉落来源确定4件套掉落率
        type_p = 1/10,  # 掉落对应套装部位的概率
        sub_stats_filter: Callable[..., bool] = None, # 设定副词条过滤器，若函数判断False直接丢掉被过滤的情况
    ) -> None:
        # 初始化道具
        self.main_stat = main_stat
        self.stats_score = stats_score
        self.p_4sub = p_4sub
        item_set = {}
        for type in ARTIFACT_TYPES:
            item_set[type] = GenshinArtifact(
                type=type,
                type_p=type_p,
                main_stat=main_stat[type],
                stats_score=stats_score,
                p_4sub=self.p_4sub,
                sub_stats_filter=sub_stats_filter,
            )
        super().__init__(item_set)

    def select_best_2piece(self, n=1) -> ScoredItem:
        """选择刷取n次后的最优2件套"""
        item_set = self.repeat(n)
        ans = select_best_combination(item_set, 2)
        return ans

    def select_best_4piece(self, n=1) -> ScoredItem:
        """选择刷取n次后的最优4件套"""
        item_set = self.repeat(n)
        ans = remove_worst_combination(item_set)
        return ans

    def get_4piece_under_condition(self, n, base_n=1500, base_p=1) -> ScoredItem:
        """
        计算在一定基础概率分布下，刷4件套+基础分布散件的分数分布
        n 表示当前刷特定本的件数
        base_n 表示其他散件可用总件数
        base_p 是散件概率调整值
        """
        base_drop_p = {}
        for type in ARTIFACT_TYPES:
            base_drop_p[type] = (
                base_p
                * (1 / 5)
                * W_MAIN_STAT[type][self.main_stat[type]]
                / dict_weight_sum(W_MAIN_STAT[type])
            )
        # 因为是按照字母序返回的列表，所以是对齐的，直接调用函数即可
        return get_mix_dist(
            self.repeat(n), self.repeat(base_n, p=base_drop_p)
        )

if __name__ == "__main__":
    pass
    """
    # 测速代码
    import time
    s = time.time()
    for i in range(10):
        get_init_state.cache_clear()
        get_state_level_up.cache_clear()
        artifact_set = GenshinArtifactSet()
        item_set_score = artifact_set.combine_set()
    t = time.time()
    print(t-s)
    """
    """
    # 基本正确性检验代码
    from matplotlib import pyplot as plt
    # flower = GenshinArtifact(type='flower')
    # plt.plot(flower.score_dist.dist, color='C1')
    # plt.plot(flower.repeat(1).score_dist.dist, color='C0')
    # plt.show()
    artifact_set = GenshinArtifactSet()
    item_set_score = artifact_set.combine_set(n=1)
    plt.plot(item_set_score.score_dist.dist, color='C1')
    plt.show()
    """
    # 测试非整数分数权重
    TEST_STAT_SCORE = {
            'hp': 0,
            'atk': 0,
            'def': 0,
            'hpp': 0,
            'atkp': 0.5,
            'defp': 0,
            'em': 0,
            'er': 0,
            'cr': 1,
            'cd': 1,
    }
    from matplotlib import pyplot as plt
    from GGanalysis.ScoredItem import check_subexp_sum
    flower = GenshinArtifact(type='flower', stats_score=TEST_STAT_SCORE)
    score = check_subexp_sum(flower, TEST_STAT_SCORE)
    print(score)
    print('atkp', flower.sub_stats_exp['atkp'])
    print('cr', flower.sub_stats_exp['cr'])
    print('cd', flower.sub_stats_exp['cd'])
    plt.plot(flower.score_dist.__dist, color='C1')
    plt.show()
    """
    # 检查4+1正确性
    # 按照年/12的方式平均，这种情况下一个月可以刷291.54件圣遗物（期望），取290为一个月的数量，3480为一年的数量
    # 但是玩家一年也不会全部体力都去刷圣遗物，取用体力的70%即一年2500件作为散件指标吧，一个版本的基准为280件。
    from matplotlib import pyplot as plt

    artifact_set = GenshinArtifactSet()
    planed_item = 290
    extra_item = 0
    item_set_score_5 = artifact_set.combine_set(n=planed_item)
    item_set_score_4plus1 = artifact_set.get_4piece_under_condition(n=planed_item)
    print(sum(item_set_score_4plus1.score_dist.dist))
    print(item_set_score_4plus1.exp)
    plt.plot(item_set_score_5.score_dist.dist, color="C0")
    plt.plot(item_set_score_4plus1.score_dist.dist, color="C1")
    plt.show()
    """
    # TODO 加入相对当前的提升，超过提升部分才纳入计算（或者说把目前的分数作为base，低于或等于这个分数的都合并到一起），即增加在现有圣遗物基础上的提升计算

    # TODO 参考ideless的方法增加筛选策略
