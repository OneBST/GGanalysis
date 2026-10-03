"""按道具优先级处理保底，并计算平稳概率与相邻获得间隔。"""

from functools import cached_property
from numbers import Integral

import numpy as np
import scipy.sparse as sp

from GGanalysis.distribution_1d import FiniteDist
from GGanalysis.markov.analysis import stationary_solve, event_count_state_dist
from GGanalysis.markov.builder import TransitionBuilder
from GGanalysis.markov.state_space import StateSpace


class PriorityPityChain:
    """按优先级处理互斥道具获取结果的有限状态保底链。

    ``item_p_list[i][j]`` 表示第 ``i`` 类道具在其保底计数重置后第 ``j`` 抽的
    原始概率。编号越小，优先级越高。``remove_pity=True`` 时，获得高优先级
    道具还会重置低优先级道具的保底计数。
    """

    def __init__(self, item_p_list, extra_state: int = 1,
                 remove_pity: bool = False):
        if not isinstance(extra_state, Integral) or extra_state < 0:
            raise ValueError("extra_state 必须是非负整数。")
        if len(item_p_list) == 0:
            raise ValueError("item_p_list 不能为空。")
        tables = tuple(np.array(table, dtype=float, copy=True) for table in item_p_list)
        for table in tables:
            if (table.ndim != 1 or len(table) < 2 or
                    not np.all(np.isfinite(table)) or
                    np.any(table < 0)):
                raise ValueError("每张保底概率表至少需要两个有限且非负的数值。")
        self.item_p_list = tables
        self.extra_state = int(extra_state)
        self.remove_pity = bool(remove_pity)
        self.item_types = len(tables)
        self.pity_state_list = tuple(len(table) + self.extra_state - 1 for table in tables)
        self.space = StateSpace.from_shape(self.pity_state_list)
        self._hit_probabilities, self._hit_targets, self.transition = self._build()

    def _build(self):
        size = self.space.N
        ids = np.arange(size, dtype=np.int64)
        states = self.space.ids_to_states(ids)
        next_miss = np.minimum(states + 1, np.asarray(self.pity_state_list) - 1)
        hit_probabilities = np.empty((self.item_types, size), dtype=float)
        hit_targets = np.empty((self.item_types, size), dtype=np.int64)
        builder = TransitionBuilder(self.space, backend="sparse")
        left = np.ones(size, dtype=float)

        for kind, table in enumerate(self.item_p_list):
            # 现有部分概率表在硬保底前会超过 1；实际获得概率受剩余概率上限约束。
            raw = table[np.minimum(states[:, kind] + 1, len(table) - 1)]
            probability = np.minimum(left, raw)
            next_states = next_miss.copy()
            next_states[:, kind] = 0
            if self.remove_pity:
                next_states[:, kind + 1:] = 0
            targets = self.space.states_to_ids(next_states)
            hit_probabilities[kind] = probability
            hit_targets[kind] = targets
            builder.add_many(ids, targets, probability)
            left -= probability

        builder.add_many(ids, self.space.states_to_ids(next_miss), left)
        return hit_probabilities, hit_targets, builder.build(check=True)

    @cached_property
    def stationary_distribution(self) -> np.ndarray:
        """随机一抽开始前，处于各完整保底状态的平稳概率。"""
        # 现有游戏模型的状态数适合稀疏直接求解；大模型改用迭代法控制内存。
        method = "direct" if self.space.N <= 10_000 else "iterative"
        return stationary_solve(self.transition, method=method)

    def stationary_rates(self) -> np.ndarray:
        """长期平稳情况下，每抽获得各类道具的综合概率。"""
        return self._hit_probabilities @ self.stationary_distribution

    def interarrival_dist(self, item_type: int, max_steps: int) -> FiniteDist:
        """平稳情况下，相邻两次获得指定类别道具的真实抽数间隔。

        ``dist[t]`` 是间隔恰好为 ``t`` 抽的概率；``tail_mass`` 是间隔超过
        ``max_steps`` 的概率。中途获得其他道具仍正常改变保底状态。
        """
        if not isinstance(item_type, Integral) or not 0 <= item_type < self.item_types:
            raise ValueError("item_type 超出道具类别范围。")
        if not isinstance(max_steps, Integral) or max_steps < 0:
            raise ValueError("max_steps 必须是非负整数。")
        hit_p = self._hit_probabilities[item_type]
        hit_to = self._hit_targets[item_type]
        stationary = self.stationary_distribution
        rate = float(hit_p @ stationary)
        if rate <= 0:
            raise ValueError("指定道具类别在平稳状态下的获得概率为零。")

        # 按目标道具的平稳命中概率加权，得到刚获得该道具后的起始状态分布。
        alive = np.bincount(
            hit_to, weights=stationary * hit_p, minlength=self.space.N
        ).astype(float) / rate
        hit = sp.coo_matrix(
            (hit_p, (hit_to, np.arange(self.space.N))),
            shape=(self.space.N, self.space.N),
        ).tocsr()
        miss = self.transition.P - hit
        miss.eliminate_zeros()

        probabilities = np.zeros(max_steps + 1, dtype=float)
        for step in range(1, max_steps + 1):
            probabilities[step] = hit_p @ alive
            alive = miss @ alive
        tail_mass = max(0.0, float(alive.sum()))
        return FiniteDist(probabilities, trim_tail_zeros=False, tail_mass=tail_mass)


def stationary_item_count_distribution(pity_p, pulls: int) -> np.ndarray:
    """长期平稳情况下，接下来 ``pulls`` 抽内获得指定道具的件数分布。

    返回数组的第 ``k`` 项是获得恰好 ``k`` 件的概率。起始保底进度按
    ``pity_p`` 对应的平稳分布混合，不代表调用者已知的当前保底进度。
    此函数只计算一类道具，独立于 ``PriorityPityChain`` 的多类别模型。
    """
    if not isinstance(pulls, Integral) or pulls < 0:
        raise ValueError("pulls 必须是非负整数。")
    chain = PriorityPityChain([pity_p], extra_state=0)
    size = chain.space.N
    hit_p = chain._hit_probabilities[0]
    hit = sp.coo_matrix(
        (hit_p, (chain._hit_targets[0], np.arange(size))),
        shape=(size, size),
    ).tocsr()
    miss = chain.transition.P - hit
    miss.eliminate_zeros()
    joint = event_count_state_dist((miss, hit), chain.stationary_distribution, pulls)
    # 保留公开返回的固定 pulls+1 长度，包括不可能数量的零概率。
    counts = joint.coeff.sum(axis=1)
    return np.pad(counts, (0, pulls + 1 - len(counts)))


__all__ = ["PriorityPityChain", "stationary_item_count_distribution"]
