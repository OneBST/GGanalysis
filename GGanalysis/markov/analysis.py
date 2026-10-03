import numpy as np
import warnings
from dataclasses import dataclass
from copy import deepcopy
from functools import cached_property
from numbers import Real
from typing import Tuple, Optional, Literal, Mapping, Sequence, Hashable
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from GGanalysis.markov.transition import (
    MarkovTransition, validate_probability_vector, prepare_event_matrices, validate_steps,
)
from GGanalysis.markov.state_space import Selector, StateSpace
from GGanalysis.state_distribution import StateDist, StateKernel

_VectorLike = Sequence[float] | np.ndarray
_MatrixLike = np.ndarray | sp.spmatrix | Sequence[Sequence[float]]
_EventInput = Sequence[_MatrixLike] | StateKernel

__all__ = [
    'first_hitting_time',
    'stationary_power',
    'stationary_eigs',
    'stationary_solve',
    'StationaryInfo',
    'StateRewards',
    'group_mass',
    'ChainAnalysis',
    'AbsorptionResult',
    'TransitionRewards',
    'event_count_state_dist',
]


def group_mass(mass: np.ndarray | Sequence[float],
               labels: np.ndarray | Sequence[Hashable]) -> dict[Hashable, float]:
    """按同形状标签汇总有限非负质量，保留标签首次出现顺序。

    不要求总质量为 1，不归一化。每个位置只能有一个可哈希标签；重叠事件
    应分别使用掩码或 ``StateRewards`` 的不同奖励行表示。
    """
    values = np.asarray(mass, dtype=float)
    keys = np.asarray(labels, dtype=object)
    if values.shape != keys.shape:
        raise ValueError("mass and labels must have the same shape.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("mass must be finite and nonnegative.")
    result = {}
    for key, value in zip(keys.flat, values.flat):
        result[key] = result.get(key, 0.0) + float(value)
    return result


class StateRewards:
    """状态条件下的一步事件次数或奖励数量的期望。

    ``space`` 定义起点编号；``weights`` 将奖励名称映射到长度为 ``space.N``
    的有限非负向量。向量元素表示从该状态出发执行一步的期望奖励。
    事件可以重叠、奖励可以超过 1，因此不要求各奖励之和为 1。
    输入被复制；公开空间及权重返回副本，构造后无需维护失效缓存。
    """

    def __init__(self, space: StateSpace,
                 weights: Mapping[str, Sequence[float] | np.ndarray]) -> None:
        self._space = deepcopy(space)
        self._names = tuple(weights)
        if any(not isinstance(name, str) for name in self._names):
            raise TypeError("reward names must be strings.")
        rows = []
        for name in self._names:
            row = np.array(weights[name], dtype=float, copy=True)
            if row.shape != (space.N,) or not np.all(np.isfinite(row)) or np.any(row < 0):
                raise ValueError(f"reward {name!r} must have {space.N} finite nonnegative weights.")
            rows.append(row)
        self._weights = np.stack(rows) if rows else np.empty((0, space.N))

    @property
    def space(self) -> StateSpace:
        """奖励起点空间的副本。"""
        return deepcopy(self._space)

    @property
    def weights(self) -> dict[str, np.ndarray]:
        """各奖励权重的独立副本。"""
        return {name: row.copy() for name, row in zip(self._names, self._weights)}

    def expectation(self, distribution: Sequence[float] | np.ndarray,
                    *, atol: float = 1e-12) -> dict[str, float]:
        """给定归一化起点分布，返回各项一步期望，不修改或归一化输入。

        传入逐抽稳态分布时结果为长期每抽奖励率；事件指示量的结果为事件率。
        ``atol`` 是总概率为 1 的绝对检查容差。条件概率需另以相应事件率为分母，
        本方法不自动推断事件包含关系。
        """
        vector = validate_probability_vector(
            distribution, self._space.N, "distribution", normalized=True, atol=atol,
        )
        return dict(zip(self._names, map(float, self._weights @ vector)))


@dataclass(frozen=True)
class StationaryInfo:
    '''平稳分布算法的收敛和误差信息。'''
    method: str
    converged: bool
    iterations: Optional[int]
    residual: float


def _finalize_stationary(
    tm: MarkovTransition,
    vector: np.ndarray,
    method: str,
    tol: float,
    iterations: Optional[int] = None,
    solver_converged: bool = True,
) -> Tuple[np.ndarray, StationaryInfo]:
    '''规范化候选向量并检查 ``P @ pi == pi``。'''
    vector = np.real_if_close(vector).real.astype(np.float64, copy=False)
    if vector.sum() < 0:
        vector = -vector
    negative_tol = max(tol * 10, 1e-14)
    if np.min(vector) < -negative_tol:
        raise RuntimeError(f"{method} returned a vector with significant negative entries.")
    vector = np.maximum(vector, 0.0)
    mass = float(vector.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise RuntimeError(f"{method} returned a zero or non-finite vector.")
    vector /= mass
    residual = float(np.linalg.norm(tm.P @ vector - vector, ord=1))
    converged = solver_converged and np.isfinite(residual) and residual <= max(tol * 10, 1e-10)
    if not converged:
        warnings.warn(
            f"{method} stationary residual is {residual:.3e}; result may be unreliable.",
            RuntimeWarning,
        )
    return vector, StationaryInfo(method, converged, iterations, residual)

def first_hitting_time(
    tm: MarkovTransition,
    init_dist: np.ndarray,
    hit_selector: Selector,
    steps: int,
    include_t0: bool = False,
    return_pos: bool = False,
    *,
    tail_tol: Optional[float] = None,
) -> Tuple[np.ndarray, float] | Tuple[np.ndarray, float, np.ndarray]:
    """
    计算指定初始分布下，首次到达 hit_selector 所定义吸收态的抽数分布。

    Parameters
    ----------
    tm : MarkovTransition
        马尔可夫转移算子。
    init_dist : np.ndarray
        长度为 tm.N 的初始概率分布向量。
    hit_selector
        定义吸收态的 selector，格式同 ``StateSpace.select_ids()``。
        例如 ``[0, None]`` 表示第一维取 0、第二维全选。
    steps : int
        最大转移步数。
    include_t0 : bool
        是否计入 t=0 的命中（初始分布落在吸收态的部分）。为 False 时，初始
        命中部分不会被删除，而是和其他初始概率一样先执行一步转移；这适合“刚从
        命中边界开始，计算下一次命中”的更新过程。
    return_pos : bool
        若为 True，额外返回 ``pos_after`` 矩阵：
        ``pos_after[t, k]`` = 在第 t 步命中时落在第 k 个吸收态的条件概率。
        默认为 False 以节省计算。
    tail_tol : float or None, optional
        非负且有限的剩余质量阈值。每次吸收命中质量后，若剩余质量不大于
        阈值则提前结束；默认 None 计算满 steps 步。启用 include_t0 时也
        检查零步吸收后的质量。提前结束只截短输出，不归一化或补全尾部；
        该阈值不保证尾部期望、方差的误差。

    Returns
    -------
    f : np.ndarray, shape (实际计算步数+1,)
        f[t] = P(首次在 t 步命中)。tail_tol=None 时长度固定为 steps+1。
    surv : float
        实际计算结束仍未命中的质量，可能包含以后命中和永不命中的质量。
    pos_after : np.ndarray, shape (实际计算步数+1, n_hit)
        命中时落在各吸收态的条件分布；仅 ``return_pos=True`` 时返回。
    """
    p = validate_probability_vector(init_dist, tm.N, "init_dist", flatten=True)
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer.")
    if tail_tol is not None and (
        isinstance(tail_tol, (bool, np.bool_)) or not isinstance(tail_tol, Real)
        or not np.isfinite(tail_tol) or tail_tol < 0
    ):
        raise ValueError("tail_tol must be None or a finite nonnegative real number.")

    space = tm.space
    f = np.zeros(steps + 1, dtype=np.float64)

    # 不需要hit位置分布，走 compile_hit_and_clear 的 block accessor
    if not return_pos:
        hitter = space.compile_hit_and_clear(hit_selector)
        if include_t0:
            f[0] = hitter(p)
            if tail_tol is not None and p.sum() <= tail_tol:
                return f[:1].copy(), float(p.sum())
        for t in range(1, steps + 1):
            p = tm(p)
            f[t] = hitter(p)
            if tail_tol is not None and p.sum() <= tail_tol:
                return f[:t + 1].copy(), float(p.sum())
        surv = float(p.sum())
        return f, surv

    # 需要返回逐位置分布
    hit_ids = space.select_ids(hit_selector)
    n_hit = len(hit_ids)

    pos_after = np.zeros((steps + 1, n_hit), dtype=np.float64)

    if include_t0:
        hit_vals = p[hit_ids]
        win_mass = float(hit_vals.sum())
        f[0] = win_mass
        if win_mass > 0:
            pos_after[0, :] = hit_vals / win_mass
        p[hit_ids] = 0.0
        if tail_tol is not None and p.sum() <= tail_tol:
            return f[:1].copy(), float(p.sum()), pos_after[:1].copy()

    for t in range(1, steps + 1):
        p = tm(p)
        hit_vals = p[hit_ids]
        win_mass = float(hit_vals.sum())
        f[t] = win_mass
        if win_mass > 0:
            pos_after[t, :] = hit_vals / win_mass
        p[hit_ids] = 0.0
        if tail_tol is not None and p.sum() <= tail_tol:
            return f[:t + 1].copy(), float(p.sum()), pos_after[:t + 1].copy()

    surv = float(p.sum())
    return f, surv, pos_after

def stationary_power(
    tm: MarkovTransition,
    tol: float = 1e-12,
    max_iter: int = 200_000,
    lazy: float = 0.0,
    x0: Optional[np.ndarray] = None,
    return_info: bool = False,
) -> np.ndarray | Tuple[np.ndarray, StationaryInfo]:
    # 幂迭代法计算平稳分布
    tm.validate()
    if tol <= 0 or max_iter <= 0:
        raise ValueError("tol and max_iter must be positive.")
    if x0 is None:
        p = np.ones(tm.N, dtype=np.float64) / tm.N
    else:
        p = validate_probability_vector(x0, tm.N, "x0", flatten=True)
        mass = float(p.sum())
        if mass == 0:
            raise ValueError("x0 must have positive total mass.")
        p /= mass

    if lazy != 0.0 and not (0.0 < lazy < 1.0):
        raise ValueError("lazy must be in (0,1)")

    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        p_next = tm._step(p)
        if lazy:
            p_next = (1.0 - lazy) * p_next + lazy * p
        s = p_next.sum()
        if s:
            p_next /= s
        difference = np.linalg.norm(p_next - p, ord=1)
        p = p_next
        if difference < tol:
            converged = True
            break

    if not converged:
        warnings.warn(f"stationary_power did not converge in {max_iter} iterations.", RuntimeWarning)
    p, info = _finalize_stationary(
        tm, p, "power", tol, iterations=iterations, solver_converged=converged
    )
    return (p, info) if return_info else p

def stationary_eigs(
    tm: MarkovTransition,
    tol: float = 1e-12,
    return_info: bool = False,
) -> np.ndarray | Tuple[np.ndarray, StationaryInfo]:
    # 使用特征值分解求平稳分布
    tm.validate()
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if tm.backend == "dense" or tm.N <= 2:
        matrix = tm.P if tm.backend == "dense" else tm.P.toarray()
        w, v = np.linalg.eig(matrix)
        k = int(np.argmin(np.abs(w - 1.0)))
        vec = np.real(v[:, k])
    else:
        # 随机矩阵的特征值 1 具有最大实部；LM 在周期链上可能选到其他单位根。
        w, v = spla.eigs(tm.P, k=1, which="LR", tol=tol)
        vec = np.real(v[:, 0])

    vec, info = _finalize_stationary(tm, vec, "eigs", tol)
    return (vec, info) if return_info else vec

def stationary_solve(
    tm: MarkovTransition,
    method: Literal["auto", "direct", "iterative"] = "auto",
    tol: float = 1e-12,
    max_iter: Optional[int] = None,
    direct_size_limit: int = 2_000,
    return_info: bool = False,
) -> np.ndarray | Tuple[np.ndarray, StationaryInfo]:
    # 解线性方程 (I - P)pi = 0 求解平稳分布，并加约束 sum(pi)=1
    tm.validate()
    if method not in ("auto", "direct", "iterative"):
        raise ValueError("method must be auto, direct or iterative.")
    if tol <= 0 or direct_size_limit <= 0:
        raise ValueError("tol and direct_size_limit must be positive.")
    if method == "auto":
        method = "direct" if tm.backend == "dense" or tm.N <= direct_size_limit else "iterative"

    iterations = None
    solver_converged = True
    if method == "direct" and tm.backend == "dense":
        A = np.eye(tm.N) - tm.P
        b = np.zeros(tm.N)
        A[-1, :] = 1.0
        b[-1] = 1.0
        pi = np.linalg.solve(A, b)
    elif method == "direct":
        I = sp.eye(tm.N, format="csc")
        A = (I - tm.P).tolil()
        A[-1, :] = 1.0
        A = A.tocsc()
        b = np.zeros(tm.N)
        b[-1] = 1.0
        pi = spla.spsolve(A, b)
    else:
        # 增广最小二乘系统不做 LU 分解，避免大型稀疏矩阵的 fill-in 内存爆炸。
        sparse_P = sp.csr_matrix(tm.P) if tm.backend == "dense" else tm.P
        A = sp.vstack(
            [sp.eye(tm.N, format="csr") - sparse_P,
             sp.csr_matrix(np.ones((1, tm.N)))],
            format="csr",
        )
        b = np.zeros(tm.N + 1)
        b[-1] = 1.0
        if max_iter is None:
            max_iter = max(1_000, tm.N * 2)
        result = spla.lsmr(A, b, atol=tol, btol=tol, maxiter=max_iter)
        pi = result[0]
        iterations = int(result[2])
        solver_converged = result[1] in (1, 2)

    pi, info = _finalize_stationary(
        tm, pi, f"solve-{method}", tol,
        iterations=iterations, solver_converged=solver_converged,
    )
    return (pi, info) if return_info else pi


def event_count_state_dist(
    event_matrices: _EventInput, initial: Sequence[float] | np.ndarray, steps: int,
    *, strategy: Literal["step", "power"] = "step", method: str = "auto",
) -> StateDist:
    """计算固定步数的累计事件数量与结束状态联合分布。

    Parameters
    ----------
    event_matrices : sequence of matrices or StateKernel
        第 r 个矩阵 K_r[end, start] 表示本步计 r 次事件及状态转移的联合概率。
        可使用稀疏矩阵；所有 K_r 之和必须列归一化。核的累计轴解释为数量。
    initial : array_like
        归一化状态初态。
    steps : int
        非负执行次数，不是累计事件数量。
    strategy : {'step', 'power'}
        step 使用稀疏矩阵批量顺序传播；power 显式转为稠密 StateKernel 并快速幂。
        大状态空间不宜选择 power；不自动选择策略。
    method : {'auto', 'direct', 'fft'}
        power 策略的卷积方法，step 策略不执行卷积后端选择。

    Returns
    -------
    StateDist
        coeff[count, state]，边缘数量分布通过 marginal_cost() 获取。
    """
    validate_steps(steps)
    if strategy not in ("step", "power") or method not in ("auto", "direct", "fft"):
        raise ValueError("invalid strategy or convolution method.")
    blocks = prepare_event_matrices(event_matrices)
    joint = validate_probability_vector(initial, blocks[0].shape[0], normalized=True, flatten=True)[None, :]
    if steps == 0:
        return StateDist(joint)
    if strategy == "power":
        kernel = StateKernel(np.stack([block.toarray() for block in blocks]))
        return kernel.apply_power(int(steps), StateDist(joint), method=method)
    active = [(r, block) for r, block in enumerate(blocks) if block.nnz]
    maximum = max(r for r, _ in active)
    for _ in range(steps):
        output = np.zeros((len(joint) + maximum, joint.shape[1]))
        for reward, block in active:
            output[reward:reward + len(joint)] += (block @ joint.T).T
        joint = output
    return StateDist(joint)


@dataclass(frozen=True)
class AbsorptionResult:
    """互斥目标的最终首次命中概率，以及永不命中其并集的概率。"""
    probabilities: dict[str, float]
    never_hit: float


class _TransientSystem:
    """固定继续状态的一次分解，同时服务不同初态和边界统计。"""
    def __init__(self, matrix: sp.csr_matrix, states: np.ndarray):
        self.states = states
        if len(states):
            self._factor = spla.splu(sp.eye(len(states), format="csc") - matrix[states][:, states].tocsc())

    def visits(self, initial: np.ndarray) -> np.ndarray:
        return self._factor.solve(initial[self.states]) if len(self.states) else np.empty(0)


class ChainAnalysis:
    """有限链的结构、最终吸收、长期平均及奖励分析。

    Parameters
    ----------
    transition : MarkovTransition
        列为起点的守恒转移。构造时复制矩阵和空间，后续外部修改不影响分析。

    Notes
    -----
    结构由严格正概率边决定。只缓存最近一次继续状态划分的线性分解，
    不构造全状态对的访问次数矩阵；不同目标可能需要重新分解。
    """
    def __init__(self, transition: MarkovTransition) -> None:
        transition.validate()
        self._space = deepcopy(transition.space)
        self._P = sp.csr_matrix(transition.P, dtype=float, copy=True)
        self._P.eliminate_zeros()
        self._forward = self._P.T.tocsr()
        self._last_system = None

    @cached_property
    def _classes(self):
        count, labels = sp.csgraph.connected_components(self._forward, directed=True, connection="strong")
        edges = self._P.tocoo()
        outgoing = np.unique(labels[edges.col[labels[edges.row] != labels[edges.col]]])
        return tuple(np.flatnonzero(labels == label) for label in range(count)
                     if label not in outgoing)

    @property
    def closed_classes(self) -> tuple[np.ndarray, ...]:
        """各闭合互通类的状态编号副本，不包含临时互通类。"""
        return tuple(states.copy() for states in self._classes)

    @staticmethod
    def _reach(graph, states, stop=()):
        seen = np.zeros(graph.shape[0], dtype=bool)
        terminal = np.zeros_like(seen)
        terminal[list(stop)] = True
        pending = list(states)
        seen[pending] = True
        while pending:
            state = pending.pop()
            if terminal[state]:
                continue
            neighbors = graph.indices[graph.indptr[state]:graph.indptr[state + 1]]
            new = neighbors[~seen[neighbors]]
            seen[new] = True
            pending.extend(new.tolist())
        return np.flatnonzero(seen)

    def reachable_from(self, states: Sequence[int]) -> np.ndarray:
        """返回从给定状态编号可达的编号，包含起点自身。"""
        states = np.asarray(states, dtype=int)
        if np.any(states < 0) or np.any(states >= self._space.N):
            raise ValueError("state IDs out of bounds.")
        return self._reach(self._forward, states)

    def _system(self, states):
        if self._last_system is None or not np.array_equal(self._last_system.states, states):
            self._last_system = _TransientSystem(self._P, states)
        return self._last_system

    def _targets(self, targets: Mapping[str, Selector]):
        groups = {name: self._space.select_ids(selector) for name, selector in targets.items()}
        ids = np.concatenate(list(groups.values())) if groups else np.empty(0, dtype=int)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("target groups must be disjoint.")
        return groups

    def _absorb(self, initial, groups):
        targets = np.concatenate(list(groups.values())) if groups else np.empty(0, dtype=int)
        continuing = np.setdiff1d(self._reach(self._P, targets), targets)
        visits = self._system(continuing).visits(initial)
        probabilities = {
            name: float(initial[ids].sum() + (self._P[ids][:, continuing] @ visits).sum())
            for name, ids in groups.items()
        }
        return AbsorptionResult(probabilities, float(1 - sum(probabilities.values())))

    def absorption(self, initial: _VectorLike, targets: Mapping[str, Selector],
                   *, include_t0: bool = True) -> AbsorptionResult:
        """计算互斥目标的竞争首次吸收；未进入任何目标的概率为 never_hit。

        initial 为归一化初态；targets 将名称映射到 StateSpace 的 selector。
        include_t0=True 计入初始命中，False 先传播一步，可计算首次返回。
        不要求原目标是吸收态，不截断无限时间，也不归一化结果。
        """
        vector = validate_probability_vector(initial, self._space.N, normalized=True, flatten=True)
        if not include_t0:
            vector = self._P @ vector
        return self._absorb(vector, self._targets(targets))

    def _class_weights(self, initial):
        groups = {str(i): states for i, states in enumerate(self._classes)}
        return np.array(list(self._absorb(initial, groups).probabilities.values()))

    @cached_property
    def _stationaries(self):
        return tuple(stationary_solve(MarkovTransition(StateSpace.from_shape([len(states)]),
                                                      self._P[states][:, states]))
                     for states in self._classes)

    def long_run_state(self, initial: _VectorLike) -> np.ndarray:
        """给定归一化初态，返回长期平均占用比例；周期链不承诺逐步分布收敛。"""
        weights = self._class_weights(validate_probability_vector(initial, self._space.N, normalized=True, flatten=True))
        result = np.zeros(self._space.N)
        for weight, states, stationary in zip(weights, self._classes, self._stationaries):
            result[states] = weight * stationary
        return result

    def expected_reward_until(self, initial: _VectorLike, targets: Mapping[str, Selector], rewards: StateRewards,
                              *, include_t0: bool = True) -> dict[str, float]:
        """直到任意目标首次命中的累计一步奖励期望。

        奖励包含进入目标的最后一步；初始命中且 include_t0=True 时奖励为零。
        要求从该初态几乎必然终止。存在非终止路径时拒绝计算，不隐式改成
        成功条件期望或进入无望区域即终止的期望。
        """
        if rewards.space.shape != self._space.shape:
            raise ValueError("reward state space does not match the chain.")
        vector = validate_probability_vector(initial, self._space.N, normalized=True, flatten=True)
        prefix = rewards.expectation(vector) if not include_t0 else {}
        if not include_t0:
            vector = self._P @ vector
        groups = self._targets(targets)
        self._absorb(vector, groups)
        targets = np.concatenate(list(groups.values())) if groups else np.empty(0, dtype=int)
        reached = self._reach(self._forward, np.flatnonzero(vector), stop=targets)
        # 用可达结构判断非终止，不把极小失败概率当作零。
        if len(np.setdiff1d(reached, np.union1d(targets, self._last_system.states))):
            raise ValueError("termination is not almost sure; specify an explicit failure target.")
        system = self._last_system
        visits = system.visits(vector)
        return {name: float(row[system.states] @ visits) + prefix.get(name, 0.0)
                for name, row in rewards.weights.items()}

    def long_run_rewards(self, rewards: "TransitionRewards", initial: _VectorLike) -> dict[str, np.ndarray | float]:
        """返回各闭合类的平均奖励率、渐近方差增长率及进入权重。

        rewards 必须描述同一转移矩阵。between_class_variance 是各类平均率
        的混合方差，即 Var(累计奖励)/n**2 的极限；非零时不能用单一线性
        方差增长率描述整体波动。每类方差用泊松方程计算，允许周期类。
        """
        if rewards._P.shape != self._P.shape or (rewards._P - self._P).nnz:
            raise ValueError("rewards must describe the same transition matrix.")
        weights = self._class_weights(validate_probability_vector(initial, self._space.N, normalized=True, flatten=True))
        means, variances = [], []
        for states, pi in zip(self._classes, self._stationaries):
            P = self._P[states][:, states]
            A, B = rewards._A[states][:, states], rewards._B[states][:, states]
            g = np.asarray(A.sum(axis=0)).ravel()
            mean = float(g @ pi)
            size = len(states)
            augmented = sp.bmat([[sp.eye(size) - P.T, np.ones((size, 1))],
                                 [pi[None, :], sp.csr_matrix((1, 1))]], format="csc")
            h = spla.spsolve(augmented, np.r_[g - mean, 0])[:size]
            # 鞅差 R-mean+h[next]-h[start] 的稳态二阶矩。
            edges = P.tocoo()
            delta = h[edges.row] - h[edges.col] - mean
            a = np.asarray(A[edges.row, edges.col]).ravel()
            b = np.asarray(B[edges.row, edges.col]).ravel()
            variance = float(pi[edges.col] @ (b + 2 * delta * a + delta**2 * edges.data))
            if variance < -1e-10:
                raise ValueError("reward moments produced a negative variance.")
            means.append(mean)
            variances.append(max(0.0, variance))  # 仅清除零附近的浮点误差。
        means = np.array(means)
        mean = float(weights @ means)
        return {"class_weights": weights, "mean_rates": means,
                "variance_rates": np.array(variances), "mean_rate": mean,
                "between_class_variance": float(weights @ (means - mean)**2)}


class TransitionRewards:
    """一个标量奖励的转移联合一阶、二阶矩，支持随机及非整数奖励。

    Parameters
    ----------
    transition : MarkovTransition
        奖励对应的状态转移。
    first, second : matrix
        first[j,i]=E[R 1(next=j)|start=i]，second 使用 R**2。
        需提供真实联合矩，不可仅用一步均值平方代替随机奖励二阶矩。
        支持稀疏输入；矩阵和转移均被复制。
    """
    def __init__(self, transition: MarkovTransition, first: _MatrixLike, second: _MatrixLike) -> None:
        transition.validate()
        self._P = sp.csr_matrix(transition.P, dtype=float, copy=True)
        self._A = sp.csr_matrix(first, dtype=float, copy=True)
        self._B = sp.csr_matrix(second, dtype=float, copy=True)
        for matrix in (self._A, self._B):
            if matrix.shape != self._P.shape or not np.all(np.isfinite(matrix.data)):
                raise ValueError("reward moment matrices must match the finite transition.")
        if np.any(self._B.data < 0):
            raise ValueError("second moments must be nonnegative.")
        # 零概率边不能带奖励；联合矩的 Cauchy 不等式是必要条件。
        if ((self._A - self._A.multiply(self._P != 0)).nnz or
                (self._B - self._B.multiply(self._P != 0)).nnz or
                np.any((self._A.multiply(self._A) - self._P.multiply(self._B)).data > 1e-12)):
            raise ValueError("reward moments are inconsistent with transition probabilities.")

    @classmethod
    def from_events(cls, event_matrices: _EventInput) -> "TransitionRewards":
        """从计数事件矩阵 K_r 构造奖励 R=r 的一阶和二阶转移矩。"""
        blocks = prepare_event_matrices(event_matrices)
        P = sum(blocks)
        return cls(MarkovTransition(StateSpace.from_shape([P.shape[0]]), P),
                   sum(r * block for r, block in enumerate(blocks)),
                   sum(r*r * block for r, block in enumerate(blocks)))

    def cumulative_moments(self, initial: _VectorLike, steps: int) -> tuple[float, float]:
        """返回固定非负步数的累计奖励 (均值, 方差)，包含全部 steps 步。

        初态归一化；只递推概率和矩，不构建完整奖励分布。奖励的条件分布
        由当前状态和下一状态确定，不额外依赖过去。
        """
        validate_steps(steps)
        p = validate_probability_vector(initial, self._P.shape[0], normalized=True, flatten=True)
        m, s = np.zeros_like(p), np.zeros_like(p)
        for _ in range(steps):
            p, m, s = (self._P @ p, self._P @ m + self._A @ p,
                       self._P @ s + 2 * (self._A @ m) + self._B @ p)
        mean = float(m.sum())
        variance = float(s.sum() - mean**2)
        if variance < -1e-10 * max(1, mean**2):
            raise RuntimeError("negative variance; check reward moments or numerical precision.")
        return mean, max(0.0, variance)
