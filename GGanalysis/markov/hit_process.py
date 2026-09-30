"""带命中标记的有限状态逐抽过程。

矩阵采用列为起点的约定。M[to, from] 是未命中转移，H[b, from]
是命中并留下边界状态 b 的概率，E[full, b] 将边界状态放回下一轮。
因此逐抽矩阵为 P = M + E @ H，周期核的第 t 层为 H @ M**(t-1) @ E。
"""

from functools import cached_property, wraps
from copy import deepcopy
from typing import Sequence, Iterator

import numpy as np
import scipy.sparse as sp

from GGanalysis.markov.analysis import stationary_solve
from GGanalysis.markov.builder import ProbabilityMatrixBuilder
from GGanalysis.markov.state_space import StateSpace
from GGanalysis.markov.transition import MarkovTransition, validate_matrix, validate_mass, validate_steps
from GGanalysis.state_distribution import StateDist, StateKernel, ConvolutionMethod


def _snapshot_property(function):
    """内部缓存独占结果，对外只提供副本。"""
    name = "_cached_" + function.__name__

    @wraps(function)
    def getter(self):
        if name not in self.__dict__:
            self.__dict__[name] = function(self)
        return deepcopy(self.__dict__[name])

    return property(getter)


class HitTransitionBuilder:
    """分别收集未命中边和命中边；命中边的终点属于边界空间。

    ``embed`` 可传 S×B 矩阵，或长度为 B 的完整状态编号数组。
    后者表示每个边界状态确定地重置到一个完整状态。
    """

    def __init__(self, full_space: StateSpace, boundary_space: StateSpace, embed):
        self.full_space = full_space
        self.boundary_space = boundary_space
        if sp.issparse(embed):
            self.embed = embed.tocsr(copy=True)
        else:
            values = np.asarray(embed)
            if values.ndim == 1:
                values = ProbabilityMatrixBuilder._ids(values, full_space.N)
                if values.size != boundary_space.N:
                    raise ValueError("embed IDs must have boundary_space.N entries.")
                self.embed = sp.csr_matrix(
                    (np.ones(boundary_space.N), (values, np.arange(boundary_space.N))),
                    shape=(full_space.N, boundary_space.N),
                )
            else:
                self.embed = sp.csr_matrix(values)
        validate_matrix(self.embed, (full_space.N, boundary_space.N), "embed")
        self.miss = ProbabilityMatrixBuilder(full_space, full_space)
        self.hit = ProbabilityMatrixBuilder(full_space, boundary_space)

    def add_miss(self, from_id: int, to_id: int, probability: float) -> None:
        self.miss.add(from_id, to_id, probability)

    def add_hit(self, from_id: int, boundary_id: int, probability: float) -> None:
        self.hit.add(from_id, boundary_id, probability)

    def add_miss_many(self, from_ids, to_ids, probabilities) -> None:
        self.miss.add_many(from_ids, to_ids, probabilities)

    def add_hit_many(self, from_ids, boundary_ids, probabilities) -> None:
        self.hit.add_many(from_ids, boundary_ids, probabilities)

    def build(self, layered: bool = False, *, check: bool = False) -> "HitProcess":
        """生成独立过程；``check=True`` 检查守恒及分层约束，不修复。"""
        process = HitProcess(
            self.full_space, self.boundary_space, self.miss.build(),
            self.hit.build(), self.embed, layered=layered,
        )
        if check:
            process.validate()
        return process


class HitProcess:
    """一次抽取至多命中一个目标，规则不随抽取时间改变的过程。

    ``layered=True`` 适用于完整状态按 ``(连续未命中次数, 边界状态)`` 排列、
    未命中只进入下一层、命中后由 E 回到第 0 层的过程。此时计算核只传播
    一个边界状态层，不传播整个完整状态空间。
    """

    @classmethod
    def from_layers(
        cls,
        full_space: StateSpace,
        boundary_space: StateSpace,
        hit_layers: Sequence[np.ndarray | sp.spmatrix],
        miss_layers: Sequence[np.ndarray | sp.spmatrix],
        embed: Sequence[int] | np.ndarray | sp.spmatrix,
    ) -> "HitProcess":
        """从每层局部矩阵自动组装并验证分层过程。

        完整状态必须按 ``(层, 边界状态)`` 连续排列。两个矩阵序列长度均为
        层数，每个矩阵形状均为 ``(B, B)``，列是本层起点；命中矩阵的行是
        边界终点，未命中矩阵的行是下一层状态。最后一层未命中必须为零。
        相同规则的层可以重复引用同一个矩阵，构建结果不与输入共享存储。

        ``embed`` 显式指定边界到完整空间的重置，可传矩阵或完整状态编号数组，
        且必须回到第零层。返回 ``layered=True`` 的过程，不自动修复概率。
        """
        size = boundary_space.N
        count, remainder = divmod(full_space.N, size)
        if remainder or len(hit_layers) != count or len(miss_layers) != count:
            raise ValueError("layer counts must match full_space.N / boundary_space.N.")
        builder = HitTransitionBuilder(full_space, boundary_space, embed)
        for layer, (hit, miss) in enumerate(zip(hit_layers, miss_layers)):
            hit = sp.coo_matrix(hit, dtype=float, copy=True)
            miss = sp.coo_matrix(miss, dtype=float, copy=True)
            validate_matrix(hit, (size, size), "hit layer")
            validate_matrix(miss, (size, size), "miss layer")
            builder.add_hit_many(layer * size + hit.col, hit.row, hit.data)
            if layer + 1 == count:
                if np.any(miss.data != 0):
                    raise ValueError("the last layer must have zero miss probability.")
            else:
                builder.add_miss_many(layer * size + miss.col,
                                      (layer + 1) * size + miss.row, miss.data)
        # 使用 cls 保持类方法的构造语义；边收集复用公共构建器。
        process = cls(full_space, boundary_space, builder.miss.build(),
                      builder.hit.build(), builder.embed, layered=True)
        process.validate()
        return process

    def __init__(self, full_space: StateSpace, boundary_space: StateSpace,
                 miss, hit, embed, layered: bool = False) -> None:
        self._full_space = deepcopy(full_space)
        self._boundary_space = deepcopy(boundary_space)
        self._miss = sp.csr_matrix(miss, dtype=float, copy=True)
        self._hit = sp.csr_matrix(hit, dtype=float, copy=True)
        self._embed = sp.csr_matrix(embed, dtype=float, copy=True)
        self._layered = bool(layered)
        size, boundary = full_space.N, boundary_space.N
        validate_matrix(self._miss, (size, size), "miss")
        validate_matrix(self._hit, (boundary, size), "hit")
        validate_matrix(self._embed, (size, boundary), "embed")
        if self.layered and size % boundary:
            raise ValueError("layered full space must contain complete boundary layers.")

    @property
    def full_space(self) -> StateSpace:
        return deepcopy(self._full_space)

    @property
    def boundary_space(self) -> StateSpace:
        return deepcopy(self._boundary_space)

    @property
    def layered(self) -> bool:
        return self._layered

    @property
    def miss(self) -> sp.csr_matrix:
        """未命中矩阵副本；修改副本不改变过程。"""
        return self._miss.copy()

    @property
    def hit(self) -> sp.csr_matrix:
        """命中矩阵副本。"""
        return self._hit.copy()

    @property
    def embed(self) -> sp.csr_matrix:
        """边界重置矩阵副本。"""
        return self._embed.copy()

    def validate(self, atol: float = 1e-12) -> None:
        """检查联合列质量、重置列质量，以及启用的分层结构，不修改数据。

        ``atol`` 仅用于概率守恒；分层结构要求禁止位置严格为零。
        """
        validate_mass(np.asarray(self._miss.sum(axis=0) + self._hit.sum(axis=0)).ravel(), 1, atol)
        validate_mass(np.asarray(self._embed.sum(axis=0)).ravel(), 1, atol)
        if self.layered:
            size = self.boundary_space.N
            rows, cols = self._miss.nonzero()
            if np.any(rows // size != cols // size + 1):
                raise ValueError("layered misses must enter exactly the next layer.")
            if np.any(self._embed.nonzero()[0] >= size):
                raise ValueError("layered embed must reset to layer zero.")

    @_snapshot_property
    def transition(self) -> MarkovTransition:
        """完整逐抽转移的独立副本，不暴露过程内部缓存。"""
        return MarkovTransition(self.full_space, self._miss + self._embed @ self._hit)

    @property
    def layer_matrices(self) -> tuple:
        """每层命中和未命中矩阵的独立副本。"""
        return deepcopy(self._layer_matrices)

    @cached_property
    def _layer_matrices(self):
        """每层的 (命中, 未命中) 矩阵，形状均为 B×B。"""
        if not self.layered:
            raise ValueError("layer_matrices requires layered=True.")
        self.validate()
        size = self.boundary_space.N
        count = self.full_space.N // size
        hits, misses = [], []
        for layer in range(count):
            cols = slice(layer * size, (layer + 1) * size)
            hits.append(self._hit[:, cols].tocsr())
            rows = slice((layer + 1) * size, (layer + 2) * size)
            misses.append(
                self._miss[rows, cols].tocsr() if layer + 1 < count
                else sp.csr_matrix((size, size))
            )
        return tuple(hits), tuple(misses)

    def first_hit(self, initial, max_steps: int, initial_layer: int | None = None,
                  *, return_remaining: bool = False) -> StateDist | tuple[StateDist, float]:
        """首次命中的联合分布，第 0 层为零。

        ``initial`` 为完整空间概率向量；指定 ``initial_layer`` 时为该层向量。
        ``max_steps`` 是非负计算上限。``return_remaining=True`` 同时返回
        递推末端仍未命中的质量（可能包含永不命中），不归一化截断结果。
        """
        validate_steps(max_steps)
        if initial_layer is not None:
            validate_steps(initial_layer)
            if not self.layered or initial_layer >= self.full_space.N // self.boundary_space.N:
                raise ValueError("initial_layer requires a valid layer in a layered process.")
        initial = np.asarray(initial, dtype=float)
        size = self.boundary_space.N if initial_layer is not None else self.full_space.N
        if initial.shape != (size,) or not np.all(np.isfinite(initial)) or np.any(initial < 0):
            raise ValueError("initial must be a finite nonnegative vector of the expected size.")
        validate_mass(initial.sum(), 1)

        if self.layered and initial_layer is not None:
            hits, misses = self._layer_matrices
            alive = np.asarray(initial, dtype=float)
            coeff = np.zeros((min(max_steps, len(hits) - initial_layer) + 1, self.boundary_space.N))
            for cost, layer in enumerate(range(initial_layer, initial_layer + len(coeff) - 1), 1):
                coeff[cost] = hits[layer] @ alive
                alive = misses[layer] @ alive
        else:
            alive = np.asarray(initial, dtype=float)
            coeff = np.zeros((max_steps + 1, self.boundary_space.N))
            for cost in range(1, max_steps + 1):
                coeff[cost] = self._hit @ alive
                alive = self._miss @ alive
        result = StateDist(coeff)
        if return_remaining:
            self.validate()
            remaining = float(alive.sum())
            validate_mass(result.total_mass() + remaining, initial.sum())
            return result, remaining
        return result

    def cycle_kernel(self, max_steps: int, *, return_remaining: bool = False
                     ) -> StateKernel | tuple[StateKernel, np.ndarray]:
        """返回周期核；可选返回每个输入边界状态的剩余未命中质量。

        ``max_steps`` 为非负计算上限；``return_remaining`` 不补全尾部状态
        或尾部矩，也不自动将剩余质量传播到后续核组合。
        """
        validate_steps(max_steps)
        size = self.boundary_space.N
        coeff = np.zeros((max_steps + 1, size, size))
        if self.layered:
            hits, misses = self._layer_matrices
            alive = np.asarray(self._embed[:size, :].toarray())
            for cost in range(1, min(max_steps, len(hits)) + 1):
                coeff[cost] = hits[cost - 1] @ alive
                alive = misses[cost - 1] @ alive
        else:
            alive = self._embed
            for cost in range(1, max_steps + 1):
                coeff[cost] = (self._hit @ alive).toarray()
                alive = self._miss @ alive
        result = StateKernel(coeff)
        if return_remaining:
            self.validate()
            remaining = np.asarray(alive.sum(axis=0)).ravel()
            validate_mass(coeff.sum(axis=(0, 1)) + remaining, 1)
            return result, remaining
        return result


class HitProcessAnalysis:
    """固定过程与单周期计算上限，共享周期核及稳态结果。

    构造时不绑定初态。不同初态通过 ``first_state_dist``、``nth_state_dist``
    或 ``iter_state_dists`` 传入，共享同一周期缓存，初态结果不自动缓存。
    ``max_steps`` 是每个周期的上限，不是多次命中的累计上限。
    稳态要求已覆盖完整周期；无限尾部还须独立检查尾部矩的截断误差。
    """

    def __init__(self, process: HitProcess, max_steps: int) -> None:
        self._process = process
        validate_steps(max_steps)
        self._max_steps = max_steps

    @property
    def process(self) -> HitProcess:
        return self._process

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def first_state_dist(
        self, initial: Sequence[float] | np.ndarray,
        initial_layer: int | None = None, *, return_remaining: bool = False,
    ) -> StateDist | tuple[StateDist, float]:
        """计算指定初态的首件联合分布，可选返回仍未命中的质量。

        ``initial`` 默认是完整空间向量；分层过程指定 ``initial_layer`` 时
        是该层的边界长度向量。输入必须归一化，不修改或保存输入。
        """
        return self.process.first_hit(initial, self.max_steps, initial_layer,
                                      return_remaining=return_remaining)

    @cached_property
    def _kernel(self) -> StateKernel:
        return self.process.cycle_kernel(self.max_steps)

    @property
    def kernel(self) -> StateKernel:
        """共享周期核的独立副本。"""
        return StateKernel(self._kernel)

    def iter_state_dists(
        self, count: int, initial: Sequence[float] | np.ndarray,
        initial_layer: int | None = None, *, method: ConvolutionMethod = "auto",
    ) -> Iterator[StateDist]:
        """依次产生第 1 至 count 件的累计联合分布，不含零目标项。

        ``count`` 为正整数，其余参数同首件计算；``method`` 控制花费卷积。
        返回迭代器以避免存储全部中间分布。已交付结果与后续递推隔离。
        """
        validate_steps(count)
        if count == 0:
            raise ValueError("count must be positive.")
        if method not in ("auto", "direct", "fft"):
            raise ValueError("method must be auto, direct or fft.")
        joint = self.first_state_dist(initial, initial_layer)
        for index in range(count):
            if index:
                joint = self._kernel.apply(joint, method=method)
            yield StateDist(joint)

    def nth_state_dist(
        self, count: int, initial: Sequence[float] | np.ndarray,
        initial_layer: int | None = None, *, method: ConvolutionMethod = "auto",
    ) -> StateDist:
        """返回指定初态下第 count 件的累计抽数与边界状态联合分布。

        参数同 ``iter_state_dists``。少量目标顺序应用核，避免矩阵核自乘的
        中间开销；较多目标使用 ``StateKernel.apply_power``。
        """
        validate_steps(count)
        if count == 0:
            raise ValueError("count must be positive.")
        if method not in ("auto", "direct", "fft"):
            raise ValueError("method must be auto, direct or fft.")
        # 将原模型的少量目标策略集中到通用分析层。
        if count <= 12:
            for result in self.iter_state_dists(count, initial, initial_layer, method=method):
                pass
            return result
        first = self.first_state_dist(initial, initial_layer)
        return self._kernel.apply_power(int(count) - 1, first, method=method)

    @_snapshot_property
    def post_hit_stationary(self) -> np.ndarray:
        """命中后边界状态的平稳分布。"""
        self.process.validate()
        self._kernel.validate()
        matrix = self._kernel.marginal_transition()
        chain = MarkovTransition(self.process.boundary_space, matrix)
        return stationary_solve(chain)

    @_snapshot_property
    def steady_state_dist(self) -> StateDist:
        """平稳条件下，一个周期的 (抽数, 命中后状态) 联合分布。"""
        return self._kernel.apply(StateDist(self.post_hit_stationary[None, :]))

    @_snapshot_property
    def steady_dist(self):
        return self.steady_state_dist.marginal_cost()

    @_snapshot_property
    def steady_full_state(self) -> np.ndarray:
        """随机观察一抽之前，完整状态所处位置的平稳概率。"""
        alive = np.asarray(self.process.embed @ self.post_hit_stationary).reshape(-1)
        occupation = np.zeros(self.process.full_space.N)
        miss = self.process.miss
        for _ in range(self.max_steps):
            occupation += alive
            alive = miss @ alive
        return occupation / occupation.sum()

    @_snapshot_property
    def long_run_hit_rate(self) -> float:
        return 1 / self.steady_dist.exp


__all__ = ["HitTransitionBuilder", "HitProcess", "HitProcessAnalysis"]
