"""按道具优先级处理保底，并计算平稳概率与相邻获得间隔。"""

from functools import cached_property
from numbers import Integral
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from GGanalysis.distribution_1d import FiniteDist
from GGanalysis.markov.analysis import stationary_solve, event_count_state_dist
from GGanalysis.markov.builder import TransitionBuilder
from GGanalysis.markov.state_space import StateSpace


class PriorityPityChain:
    """按优先级处理互斥道具获取结果的有限状态保底链。

    Parameters
    ----------
    item_p_list : sequence of probability tables
        item_p_list[i][j] 为第 i 类在计数重置后第 j 抽的原始概率，索引 0
        为占位。类别编号越小优先级越高，实际概率截到尚未分配的剩余概率。
    extra_state : int
        非负，默认 1。每类状态数为 len(table)+extra_state-1；超出表的
        计数继续使用末项概率。没有额外计数需求时可显式设为 0。
    remove_pity : bool
        默认 False。True 时命中高优先级类别也重置所有低优先级计数；
        False 时只重置本类别，其他类别的未获得计数继续增加。

    Notes
    -----
    构造器按上述规则自动构造完整转移。与普通边构造器不同，不需再添加边。
    默认数量查询观察随机一抽之前的平稳状态；间隔查询观察刚命中后的状态。
    原始表达到 1 不保证低优先级本抽必得，因为高优先级可占用概率。
    构造后不要修改概率表、转移或规则参数，缓存的稳态不会自动失效。
    状态数 N 为各类 (len(table)+extra_state-1) 的乘积，类别增加可能使状态
    空间迅速膨胀。迭代稳态法可避免大规模 LU，但不能解决状态数本身的爆炸；
    大模型应考虑可达状态压缩或专用表示。
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
        return stationary_solve(self.transition, method="auto", direct_size_limit=10_000)

    def stationary_rates(self) -> np.ndarray:
        """长期平稳情况下，每抽获得各类道具的综合概率。"""
        return self._hit_probabilities @ self.stationary_distribution

    def _event_matrices(self, item_type: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """拆出指定类别的未命中/命中转移，其他类别仍按完整规则改变状态。"""
        if not isinstance(item_type, Integral) or not 0 <= item_type < self.item_types:
            raise ValueError("item_type 超出道具类别范围。")
        hit = sp.csr_matrix(
            (self._hit_probabilities[item_type], (self._hit_targets[item_type], np.arange(self.space.N))),
            shape=(self.space.N, self.space.N),
        )
        hit.eliminate_zeros()
        miss = self.transition.P - hit
        miss.eliminate_zeros()
        return miss, hit

    def item_num_dist(self, item_type: int, pull: int,
                      initial: Sequence[float] | np.ndarray | None = None) -> FiniteDist:
        """返回固定 pull 抽内获得指定类别的数量分布，长度为 pull+1。

        item_type 是从零开始的类别编号。initial 为完整状态空间的归一化
        初态，默认使用随机一抽开始前的平稳分布；不使用命中后起点。
        其他类别仍正常改变保底进度，返回 FiniteDist[k]=恰好获得 k 件的概率。
        """
        miss, hit = self._event_matrices(item_type)
        if initial is None:
            initial = self.stationary_distribution
        joint = event_count_state_dist((miss, hit), initial, pull)
        counts = joint.coeff.sum(axis=1)
        return FiniteDist(np.pad(counts, (0, pull+1-len(counts))), trim_tail_zeros=False)

    def interarrival_dist(self, item_type: int, max_steps: int) -> FiniteDist:
        """平稳情况下，相邻两次获得指定类别道具的真实抽数间隔。

        ``dist[t]`` 是间隔恰好为 ``t`` 抽的概率；``tail_mass`` 是间隔超过
        ``max_steps`` 的概率。中途获得其他道具仍正常改变保底状态。
        """
        miss, hit = self._event_matrices(item_type)
        if not isinstance(max_steps, Integral) or max_steps < 0:
            raise ValueError("max_steps 必须是非负整数。")
        hit_p = self._hit_probabilities[item_type]
        stationary = self.stationary_distribution
        rate = float(hit_p @ stationary)
        if rate <= 0:
            raise ValueError("指定道具类别在平稳状态下的获得概率为零。")

        # 按目标道具的平稳命中概率加权，得到刚获得该道具后的起始状态分布。
        alive = np.asarray(hit @ stationary).ravel() / rate

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
    此函数是构造单类别 PriorityPityChain 并调用 item_num_dist 的便捷入口。
    多类别模型请直接调用链的 item_num_dist，可保留其他类别对保底的影响。
    """
    if not isinstance(pulls, Integral) or pulls < 0:
        raise ValueError("pulls 必须是非负整数。")
    chain = PriorityPityChain([pity_p], extra_state=0)
    return chain.item_num_dist(0, pulls).dist


__all__ = ["PriorityPityChain", "stationary_item_count_distribution"]
