from copy import deepcopy
from functools import lru_cache
from typing import Callable

import numpy as np
import math
import itertools

from GGanalysis.games.honkai_star_rail.relic_data import *
from GGanalysis.ScoredItem.genshin_like_scored_item import *
from GGanalysis.ScoredItem.scored_item import ScoredItem, ScoredItemSet

"""
    崩坏：星穹铁道遗器类
"""
__all__ = [
    "StarRailRelic",
    "StarRailRelicSet",
    "StarRailCavernRelics",
    "StarRailPlanarOrnaments",
    "RELIC_TYPES",
    "CAVERN_RELICS",
    "PLANAR_ORNAMENTS",
    "STAT_NAME",
    "W_MAIN_STAT",
    "W_SUB_STAT",
    "DEFAULT_MAIN_STAT",
    "DEFAULT_STAT_SCORE",
    "DEFAULT_STAT_COLOR",
]

# 副词条档位
SUB_STATS_RANKS = [8, 9, 10]
# 权重倍数乘数，必须为整数，越大计算越慢精度越高
RANK_MULTI = 1
# 全局遗器副词条权重
STATS_WEIGHTS = {}

get_init_state = create_get_init_state(STATS_WEIGHTS, SUB_STATS_RANKS, RANK_MULTI)
get_state_level_up = create_get_state_level_up(STATS_WEIGHTS, SUB_STATS_RANKS, RANK_MULTI)

def set_using_weight(new_weight: dict):
    """更换采用权重时要刷新缓存，注意得分权重必须小于等于1"""
    global STATS_WEIGHTS
    global get_init_state
    global get_state_level_up
    STATS_WEIGHTS = new_weight
    # print('Refresh weight cache!')
    get_init_state = create_get_init_state(STATS_WEIGHTS, SUB_STATS_RANKS, RANK_MULTI)
    get_state_level_up = create_get_state_level_up(STATS_WEIGHTS, SUB_STATS_RANKS, RANK_MULTI)

def dict_weight_sum(weights: dict):
    """获得字典中所有值的和"""
    return sum(weights.values())

def get_combinations_p(stats_p: dict, select_num=4):
    """计算获得拥有4个副词条的五星遗器不同副词条组合的概率"""
    ans = {}
    weight_all = dict_weight_sum(stats_p)
    for perm in itertools.permutations(list(stats_p.keys()), select_num):
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

class StarRailRelic(ScoredItem):
    """崩铁遗器类"""
    def __init__(
        self,
        type: str = "hands",  # 道具类型
        type_p = 1/4,  # 每次获得道具是是类型道具概率
        main_stat: str = None,  # 主词条属性
        sub_stats_select_weight: dict = W_SUB_STAT,  # 副词条抽取权重
        main_stat_score: dict = DEFAULT_MAIN_STAT_SCORE,  # 主词条默认计算属性
        stats_score: dict = DEFAULT_STAT_SCORE,  # 词条评分权重
        p_4sub: float = P_INIT4_DROP,  # 初始4词条掉落率
        sub_stats_filter: Callable[..., bool] = None, # 设定副词条过滤器，若函数判断False直接丢掉被过滤的情况
        forced_combinations = None, # 直接设定初始词条组合
    ) -> None:
        # 计算获得主词条概率
        self.type = type
        self.main_stat_score = main_stat_score
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
        # 遍历所有情况累加，满级15级，每3级强化一次，初始4可以强化5次
        for stat_comb in list(self.sub_stats_combinations.keys()):
            if sub_stats_filter is not None:
                if sub_stats_filter(stat_comb) is False:
                    continue
            temp_base = get_init_state(stat_comb, init_score=self.main_stat_score[self.main_stat])
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

class StarRailRelicSet(ScoredItemSet):
    def __init__(
        self,
        main_stat: dict = DEFAULT_MAIN_STAT,
        main_stat_score: dict = DEFAULT_MAIN_STAT_SCORE,
        stats_score: dict = DEFAULT_STAT_SCORE,
        set_types: list = CAVERN_RELICS,
        p_4sub: float = P_INIT4_DROP,  # 根据圣遗物掉落来源确定4件套掉落率
        type_p = 1/8,  # 掉落对应套装部位的概率
        sub_stats_filter: Callable[..., bool] = None, # 设定副词条过滤器，若函数判断False直接丢掉被过滤的情况
    ) -> None:
        # 初始化道具
        self.main_stat = main_stat
        self.main_stat_score = main_stat_score
        self.stats_score = stats_score
        self.p_4sub = p_4sub
        self.set_types = set_types
        item_set = {}
        for type in set_types:
            item_set[type] = StarRailRelic(
                type=type,
                type_p=type_p,
                main_stat=main_stat[type],
                main_stat_score=main_stat_score,
                stats_score=stats_score,
                p_4sub=self.p_4sub,
                sub_stats_filter=sub_stats_filter,
            )
        super().__init__(item_set)

class StarRailCavernRelics(StarRailRelicSet):
    # 崩铁遗器四件套
    def __init__(
        self,
        main_stat: dict = DEFAULT_MAIN_STAT,
        main_stat_score: dict = DEFAULT_MAIN_STAT_SCORE,
        stats_score: dict = DEFAULT_STAT_SCORE,
        set_types: list = CAVERN_RELICS,
        p_4sub: float = P_INIT4_DROP,  # 根据圣遗物掉落来源确定4件套掉落率
        type_p = 1/8,  # 掉落对应套装部位的概率
        sub_stats_filter: Callable[..., bool] = None, # 设定副词条过滤器，若函数判断False直接丢掉被过滤的情况
    ) -> None:
        super().__init__(
            main_stat = main_stat,
            main_stat_score = main_stat_score,
            stats_score = stats_score,
            set_types = set_types,
            p_4sub = p_4sub,
            type_p = type_p,
            sub_stats_filter = sub_stats_filter, 
        )

class StarRailPlanarOrnaments(StarRailRelicSet):
    # 崩铁遗器两件套
    def __init__(
        self,
        main_stat: dict = DEFAULT_MAIN_STAT,
        main_stat_score: dict = DEFAULT_MAIN_STAT_SCORE,
        stats_score: dict = DEFAULT_STAT_SCORE,
        set_types: list = PLANAR_ORNAMENTS,
        p_4sub: float = P_INIT4_DROP,  # 根据圣遗物掉落来源确定4件套掉落率
        type_p = 1/4,  # 掉落对应套装部位的概率
        sub_stats_filter: Callable[..., bool] = None, # 设定副词条过滤器，若函数判断False直接丢掉被过滤的情况
    ) -> None:
        super().__init__(
            main_stat = main_stat,
            main_stat_score = main_stat_score,
            stats_score = stats_score,
            set_types = set_types,
            p_4sub = p_4sub,
            type_p = type_p,
            sub_stats_filter = sub_stats_filter, 
        )

if __name__ == "__main__":
    pass