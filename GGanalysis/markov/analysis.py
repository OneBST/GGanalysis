import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from GGanalysis.markov.transition import MarkovTransition
from GGanalysis.markov.state_space import Selector

__all__ = [
    'first_hitting_time',
    'stationary_power',
    'stationary_eigs',
    'stationary_solve',
    'StationaryInfo',
]


@dataclass(frozen=True)
class StationaryInfo:
    '''平稳分布算法的收敛和误差信息。'''
    method: str
    converged: bool
    iterations: Optional[int]
    residual: float


def _validate_probability_vector(vector: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
    if vector.size != size:
        raise ValueError(f"{name} must have length {size}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(vector < 0):
        raise ValueError(f"{name} must not contain negative probabilities.")
    return vector


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

    Returns
    -------
    f : np.ndarray, shape (steps+1,)
        f[t] = P(首次在 t 步命中)。
    surv : float
        在 0..steps 步内始终未命中的概率（即停留在瞬态的概率）。
    pos_after : np.ndarray, shape (steps+1, n_hit)
        命中时落在各吸收态的条件分布；仅 ``return_pos=True`` 时返回。
    """
    p = _validate_probability_vector(init_dist, tm.N, "init_dist")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer.")

    space = tm.space
    f = np.zeros(steps + 1, dtype=np.float64)

    # 不需要hit位置分布，走 compile_hit_and_clear 的 block accessor
    if not return_pos:
        hitter = space.compile_hit_and_clear(hit_selector)
        if include_t0:
            f[0] = hitter(p)
        for t in range(1, steps + 1):
            p = tm(p)
            f[t] = hitter(p)
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

    for t in range(1, steps + 1):
        p = tm(p)
        hit_vals = p[hit_ids]
        win_mass = float(hit_vals.sum())
        f[t] = win_mass
        if win_mass > 0:
            pos_after[t, :] = hit_vals / win_mass
        p[hit_ids] = 0.0

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
    if tol <= 0 or max_iter <= 0:
        raise ValueError("tol and max_iter must be positive.")
    if x0 is None:
        p = np.ones(tm.N, dtype=np.float64) / tm.N
    else:
        p = _validate_probability_vector(x0, tm.N, "x0")
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
