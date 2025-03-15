from GGanalysis.ScoredItem.scored_item import ScoredItem
from functools import lru_cache
import itertools
import numpy as np
from typing import Callable

def create_get_init_state(stats_weights=None, sub_stats_ranks=[7,8,9,10], rank_multi=1) -> Callable[[list[str], float], ScoredItem]:
    """返回计算 获得拥有指定初始副词条的道具初始得分分布，及该得分下每个副词条的期望占比 的函数"""
    @lru_cache(maxsize=65536)
    def get_init_state(stat_comb, default_weight=0):
        score_dist = np.zeros(40 * rank_multi + 1)
        sub_stat_exp = {}
        for sub_stat in stat_comb:
            sub_stat_exp[sub_stat] = np.zeros(40 * rank_multi + 1)
        # 生成所有可能的初始副词条组合
        stat_score_combinations = itertools.product(sub_stats_ranks, repeat=len(stat_comb))
        for stat_score in stat_score_combinations:
            total_score = 0
            for score, stat in zip(stat_score, stat_comb):
                total_score += score * stats_weights.get(stat, default_weight) * rank_multi
            # 采用比例分配
            L = int(total_score)
            R = L + 1
            w_L = R - total_score
            w_R = total_score - L
            R = min(R, 40 * rank_multi)
            score_dist[L] += w_L
            score_dist[R] += w_R
            for score, stat in zip(stat_score, stat_comb):
                sub_stat_exp[stat][L] += score * w_L
                sub_stat_exp[stat][R] += score * w_R
        # 对于所有种情况进行归一化 并移除末尾的0，节省一点后续计算
        for sub_stat in stat_comb:
            sub_stat_exp[sub_stat] = np.divide(
                sub_stat_exp[sub_stat],
                score_dist,
                out=np.zeros_like(sub_stat_exp[sub_stat]),
                where=score_dist != 0,
            )
            sub_stat_exp[sub_stat] = np.trim_zeros(sub_stat_exp[sub_stat], "b")
        score_dist /= len(sub_stats_ranks) ** len(stat_comb)
        score_dist = np.trim_zeros(score_dist, "b")
        return ScoredItem(score_dist, sub_stat_exp)
    return get_init_state

def create_get_state_level_up(stats_weights=None, sub_stats_ranks=[7,8,9,10], rank_multi=1) -> Callable:
    """返回计算 这个函数计算在给定选择词条下升1级的分数分布及每个分数下不同副词条贡献期望占比 的函数"""
    @lru_cache(maxsize=65536)
    def get_state_level_up(stat_comb, default_weight=0):
        score_dist = np.zeros(10 * rank_multi + 1)
        sub_stat_exp = {}
        for stat in stat_comb:
            sub_stat_exp[stat] = np.zeros(10 * rank_multi + 1)
        # 枚举升级词条及词条数，共4*4=16种 (如果stat_comb数量不为4则有不同)
        for stat in stat_comb:
            for j in sub_stats_ranks:
                score = stats_weights.get(stat, default_weight) * j * rank_multi
                # 采用比例分配
                L = int(score)
                R = L + 1
                w_L = R - score
                w_R = score - L
                R = min(R, 10 * rank_multi)
                # 记录数据
                score_dist[L] += w_L
                sub_stat_exp[stat][L] += j * w_L
                score_dist[R] += w_R
                sub_stat_exp[stat][R] += j * w_R
        # 对于各种组合情况进行归一化 并移除末尾的0，节省一点后续计算
        for stat in stat_comb:
            sub_stat_exp[stat] = np.divide(
                sub_stat_exp[stat],
                score_dist,
                out=np.zeros_like(sub_stat_exp[stat]),
                where=score_dist != 0,
            )
            sub_stat_exp[stat] = np.trim_zeros(sub_stat_exp[stat], "b")
        score_dist /= len(sub_stats_ranks) * len(stat_comb)
        score_dist = np.trim_zeros(score_dist, "b")
        return ScoredItem(score_dist, sub_stat_exp)
    return get_state_level_up