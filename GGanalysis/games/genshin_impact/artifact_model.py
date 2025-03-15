from copy import deepcopy
from typing import Callable

import numpy as np
import warnings
import itertools

from GGanalysis import FiniteDist
from GGanalysis.games.genshin_impact.artifact_data import *
from GGanalysis.ScoredItem.genshin_like_scored_item import *
from GGanalysis.ScoredItem.scored_item import ScoredItem, ScoredItemSet

"""
    原神圣遗物类
    部分代码修改自 https://github.com/ideless/reliq
"""
__all__ = [
    "GenshinArtifact",
    "GenshinDefinedArtifact",
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
SUB_STATS_RANKS = [7, 8, 9, 10]
# 权重倍数乘数，必须为整数，越大计算越慢精度越高
RANK_MULTI = 1
# 全局圣遗物副词条权重
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
    """获得拥有4个副词条的五星圣遗物不同副词条组合的概率"""
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
            # 检查主词条冲突
            if self.main_stat not in W_MAIN_STAT[type].keys():
                warnings.warn(f"The main_stat is set incorrectly, {type} can't have {main_stat} as main_stat.", UserWarning)
        drop_p = (
            type_p
            * W_MAIN_STAT[self.type][self.main_stat]
            / dict_weight_sum(W_MAIN_STAT[self.type])
        )
        # 确定可选副词条
        self.sub_stats_weight = deepcopy(sub_stats_select_weight)
        if self.main_stat in self.sub_stats_weight:
            del self.sub_stats_weight[self.main_stat]
        # 确定初始拥有四条副词条概率
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

def defined_enhancement_dist(time=5, pity=2, p=0.5):
    '''
        计算自定义圣遗物强化满级后命中自定义副词条次数分布
        p表示命中选定属性概率, time表示强化次数, pity表示选定属性至少命中多少次
        强化规则是，强化到剩余强化次数=max(至少命中次数-选中属性命中次数,0) 时，接下来都命中选定属性
    '''
    # M中记录当前强化命中次数分布（任意选定属性）
    M = np.zeros((2, time+1), dtype=float)
    M[0, 0] = 1
    # 递推进行计算
    for i in range(0, time):
        for j in range(0, time):
            if time-i+j <= pity:
                # 当命中次数+剩余次数小于等于指定值，触发保底
                M[(i+1)%2, j+1] += M[i%2, j]
            else:
                M[(i+1)%2, j+1] += p * M[i%2, j]
                M[(i+1)%2, j] += (1-p) * M[i%2, j]
        # 清除滚动数组当前状态
        M[i%2, :] = 0
    return FiniteDist(M[time%2, :])

class GenshinDefinedArtifact(ScoredItem):
    """
        原神「祝圣之霜」自定义圣遗物类
        使用「祝圣之霜」(Sanctifying Elixir)定义道具主词条及两种副词条后，有特别的副词条强化规则。强化时至少强化两次选定的副词条。
        推测具体逻辑为：如果当前剩余强化次数与已命中选定追加属性次数之和未达到保底的命中次数，则接下来的强化必定命中选定词条。
        推测定义得到初始4词条概率为34%，等同周本/合成台/深渊盒子
    """
    def __init__(
        self,
        type: str = "flower",  # 定义道具类型
        main_stat: str = None,  # 定义主词条属性
        select_sub_stats: list[str] = None,  # 定义副词条属性
        sub_stats_select_weight: dict = W_SUB_STAT,  # 副词条抽取权重
        stats_score: dict = DEFAULT_STAT_SCORE,  # 词条评分权重
        p_4sub: float = P_DROP_STATE['converted_by_alchemy_table'],  # 根据圣遗物掉落来源确定4件套掉落率
    ) -> None:
        # 设定主词条
        self.type = type
        if main_stat is not None:
            self.main_stat = main_stat
            # 检查主词条冲突
            if self.main_stat not in W_MAIN_STAT[type].keys():
                warnings.warn(f"The main_stat is set incorrectly, {type} can't have {main_stat} as main_stat.", UserWarning)
        else:
            self.main_stat = DEFAULT_MAIN_STAT[self.type]
        # 检查副词条冲突
        if select_sub_stats is None:
            raise ValueError("Two sub_stats must be provided in select_sub_stats.")
        else:
            if main_stat in select_sub_stats:
                raise ValueError("main_stats cannot overlap with sub_stats.")
        self.select_sub_stats = select_sub_stats
        # 确定剩余可选副词条
        self.sub_stats_weight = deepcopy(sub_stats_select_weight)
        if self.main_stat in self.sub_stats_weight:
            del self.sub_stats_weight[self.main_stat]
        for select_sub_stat in self.select_sub_stats:
            del self.sub_stats_weight[select_sub_stat]
        # 确定初始拥有四条副词条概率
        self.p_4sub = p_4sub
        # 词条权重改变时应清除缓存
        self.stats_score = stats_score
        if self.stats_score != STATS_WEIGHTS:
            set_using_weight(self.stats_score)
        # 计算选择剩余两个副词条组合概率
        self.sub_stats_combinations = get_combinations_p(stats_p=self.sub_stats_weight, select_num=2)
        ans = ScoredItem()
        enhancement_5 = defined_enhancement_dist(time=5, pity=2)
        enhancement_4 = defined_enhancement_dist(time=4, pity=2)
        def calc_level_up(base, level_up, time):
            temp = base
            for i in range(time):
                temp = temp * level_up
            return temp
        # 将指定词条和非指定词条拆开，对于每种词条分配情况分别进行计算后加权合并
        for stat_comb in list(self.sub_stats_combinations.keys()):
            # 枚举另外两个词条属性
            defined_base = get_init_state(tuple(self.select_sub_stats))
            random_base = get_init_state(stat_comb)
            defined_level_up = get_state_level_up(tuple(self.select_sub_stats))
            random_level_up = get_state_level_up(stat_comb)
            ans_5 = ScoredItem()
            for i in range(5+1):
                defined_ans = calc_level_up(defined_base, defined_level_up, i)
                random_ans = calc_level_up(random_base, random_level_up, 5-i)
                ans_5 += (defined_ans * random_ans) * enhancement_5[i]
            # 强化4次情况
            ans_4 = ScoredItem()
            for i in range(4+1):
                defined_ans = calc_level_up(defined_base, defined_level_up, i)
                random_ans = calc_level_up(random_base, random_level_up, 4-i)
                ans_4 += (defined_ans * random_ans) * enhancement_4[i]
            # 加权更新结果
            ans += ((1 - self.p_4sub) * ans_4 + self.p_4sub * ans_5) * self.sub_stats_combinations[stat_comb]

        super().__init__(ans.score_dist, ans.sub_stats_exp, 1, stats_score=self.stats_score)

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
    plt.plot(flower.score_dist.dist, color='C1')
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
