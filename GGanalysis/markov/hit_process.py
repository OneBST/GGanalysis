"""带命中标记的有限状态逐抽过程。

矩阵采用列为起点的约定。M[to, from] 是未命中转移，H[b, from]
是命中并留下边界状态 b 的概率，E[full, b] 将边界状态放回下一轮。
因此逐抽矩阵为 P = M + E @ H，周期核的第 t 层为 H @ M**(t-1) @ E。
"""

from functools import cached_property

import numpy as np
import scipy.sparse as sp

from GGanalysis.markov.analysis import stationary_solve
from GGanalysis.markov.builder import TransitionBuilder
from GGanalysis.markov.state_space import StateSpace
from GGanalysis.markov.transition import MarkovTransition
from GGanalysis.state_distribution import StateDist, StateKernel


class HitTransitionBuilder:
    """分别收集未命中边和命中边；命中边的终点属于边界空间。

    ``embed`` 可传 S×B 矩阵，或长度为 B 的完整状态编号数组。
    后者表示每个边界状态确定地重置到一个完整状态。
    """

    def __init__(self, full_space: StateSpace, boundary_space: StateSpace, embed):
        self.full_space = full_space
        self.boundary_space = boundary_space
        if sp.issparse(embed):
            self.embed = embed.tocsr()
        else:
            values = np.asarray(embed)
            if values.ndim == 1:
                self.embed = sp.csr_matrix(
                    (np.ones(boundary_space.N), (values, np.arange(boundary_space.N))),
                    shape=(full_space.N, boundary_space.N),
                )
            else:
                self.embed = sp.csr_matrix(values)
        self.miss = TransitionBuilder(full_space, backend="sparse")
        self._hit_from = []
        self._hit_to = []
        self._hit_prob = []

    def add_miss(self, from_id: int, to_id: int, probability: float) -> None:
        self.miss.add(from_id, to_id, probability)

    def add_hit(self, from_id: int, boundary_id: int, probability: float) -> None:
        self._hit_from.append(from_id)
        self._hit_to.append(boundary_id)
        self._hit_prob.append(probability)

    def add_miss_many(self, from_ids, to_ids, probabilities) -> None:
        self.miss.add_many(from_ids, to_ids, probabilities)

    def add_hit_many(self, from_ids, boundary_ids, probabilities) -> None:
        self._hit_from.extend(from_ids)
        self._hit_to.extend(boundary_ids)
        self._hit_prob.extend(probabilities)

    def build(self, layered: bool = False) -> "HitProcess":
        hit = sp.coo_matrix(
            (self._hit_prob, (self._hit_to, self._hit_from)),
            shape=(self.boundary_space.N, self.full_space.N),
        ).tocsr()
        return HitProcess(
            self.full_space, self.boundary_space, self.miss.build().P,
            hit, self.embed, layered=layered,
        )


class HitProcess:
    """一次抽取至多命中一个目标，规则不随抽取时间改变的过程。

    ``layered=True`` 适用于完整状态按 ``(连续未命中次数, 边界状态)`` 排列、
    未命中只进入下一层、命中后由 E 回到第 0 层的过程。此时计算核只传播
    一个边界状态层，不传播整个完整状态空间。
    """

    def __init__(self, full_space, boundary_space, miss, hit, embed, layered=False):
        self.full_space = full_space
        self.boundary_space = boundary_space
        self.miss = sp.csr_matrix(miss)
        self.hit = sp.csr_matrix(hit)
        self.embed = sp.csr_matrix(embed)
        self.layered = layered

    @cached_property
    def transition(self) -> MarkovTransition:
        """忽略命中标记后，完整状态上的逐抽转移矩阵。"""
        return MarkovTransition(self.full_space, self.miss + self.embed @ self.hit)

    @cached_property
    def layer_matrices(self):
        """每层的 (命中, 未命中) 矩阵，形状均为 B×B。"""
        size = self.boundary_space.N
        count = self.full_space.N // size
        hits, misses = [], []
        for layer in range(count):
            cols = slice(layer * size, (layer + 1) * size)
            hits.append(self.hit[:, cols].tocsr())
            rows = slice((layer + 1) * size, (layer + 2) * size)
            misses.append(
                self.miss[rows, cols].tocsr() if layer + 1 < count
                else sp.csr_matrix((size, size))
            )
        return tuple(hits), tuple(misses)

    def first_hit(self, initial, max_steps: int, initial_layer: int | None = None) -> StateDist:
        """首次命中的 (抽数, 边界状态) 联合分布；第 0 层恒为零。"""
        if self.layered and initial_layer is not None:
            hits, misses = self.layer_matrices
            alive = np.asarray(initial, dtype=float)
            coeff = np.zeros((min(max_steps, len(hits) - initial_layer) + 1, self.boundary_space.N))
            for cost, layer in enumerate(range(initial_layer, initial_layer + len(coeff) - 1), 1):
                coeff[cost] = hits[layer] @ alive
                alive = misses[layer] @ alive
        else:
            alive = np.asarray(initial, dtype=float)
            coeff = np.zeros((max_steps + 1, self.boundary_space.N))
            for cost in range(1, max_steps + 1):
                coeff[cost] = self.hit @ alive
                alive = self.miss @ alive
        return StateDist(coeff)

    def cycle_kernel(self, max_steps: int) -> StateKernel:
        """相邻两次命中的周期核，保留命中后的边界状态。"""
        size = self.boundary_space.N
        coeff = np.zeros((max_steps + 1, size, size))
        if self.layered:
            hits, misses = self.layer_matrices
            alive = np.asarray(self.embed[:size, :].toarray())
            for cost in range(1, min(max_steps, len(hits)) + 1):
                coeff[cost] = hits[cost - 1] @ alive
                alive = misses[cost - 1] @ alive
        else:
            alive = self.embed
            for cost in range(1, max_steps + 1):
                coeff[cost] = (self.hit @ alive).toarray()
                alive = self.miss @ alive
        return StateKernel(coeff)


class HitProcessAnalysis:
    """给定首件起点和计算上限后，复用周期核进行抽数及平稳分析。

    平稳统计要求 ``max_steps`` 已覆盖完整命中周期。有限保底过程可直接使用
    保底抽数；无有限保底的过程则需选择足够大的截断上限。
    """

    def __init__(self, process: HitProcess, initial, max_steps: int,
                 initial_layer: int | None = None):
        self.process = process
        self.initial = np.asarray(initial, dtype=float)
        self.initial_layer = initial_layer
        self.max_steps = max_steps

    @cached_property
    def first_state_dist(self) -> StateDist:
        return self.process.first_hit(self.initial, self.max_steps, self.initial_layer)

    @cached_property
    def kernel(self) -> StateKernel:
        return self.process.cycle_kernel(self.max_steps)

    def nth_state_dist(self, count: int) -> StateDist:
        """第 count 件命中时的累计抽数和边界状态。"""
        return self.kernel.apply_power(count - 1, self.first_state_dist)

    @cached_property
    def post_hit_stationary(self) -> np.ndarray:
        """命中后边界状态的平稳分布。"""
        self.kernel.validate()
        matrix = self.kernel.marginal_transition()
        chain = MarkovTransition(self.process.boundary_space, matrix)
        return stationary_solve(chain)

    @cached_property
    def steady_state_dist(self) -> StateDist:
        """平稳条件下，一个周期的 (抽数, 命中后状态) 联合分布。"""
        return self.kernel.apply(StateDist(self.post_hit_stationary[None, :]))

    @cached_property
    def steady_dist(self):
        return self.steady_state_dist.marginal_cost()

    @cached_property
    def steady_full_state(self) -> np.ndarray:
        """随机观察一抽之前，完整状态所处位置的平稳概率。"""
        alive = np.asarray(self.process.embed @ self.post_hit_stationary).reshape(-1)
        occupation = np.zeros(self.process.full_space.N)
        for _ in range(self.max_steps):
            occupation += alive
            alive = self.process.miss @ alive
        return occupation / occupation.sum()

    @cached_property
    def long_run_hit_rate(self) -> float:
        return 1 / self.steady_dist.exp


__all__ = ["HitTransitionBuilder", "HitProcess", "HitProcessAnalysis"]
