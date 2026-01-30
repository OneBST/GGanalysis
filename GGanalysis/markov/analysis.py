import numpy as np
from typing import Tuple, Optional
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from GGanalysis.markov.transition import MarkovTransition

__all__ = [
    'first_hitting_time',
    'stationary_power',
    'stationary_eigs',
    'stationary_solve',
]

def first_hitting_time(
    tm: MarkovTransition,
    p0: np.ndarray,
    hitter,
    steps: int,
    include_t0: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    物理时间对齐的首次到达时间分布

    Parameters
    ----------
    tm : MarkovTransition
        马尔可夫转移算子
    p0 : np.ndarray
        初始分布
    hitter : callable
        f(p) -> hit_prob，并会原地清零命中部分
        （通常由 StateSpace.compile_hitter 生成）
    steps : int
        最大物理时间
    include_t0 : bool
        是否计入 t=0 的命中

    Returns
    -------
    f : np.ndarray
        f[t] = P(首次在 t 命中)
    surv : float
        在 0..steps 内始终未命中的概率
    """
    p = np.asarray(p0, dtype=np.float64).reshape(-1).copy()
    if p.size != tm.N:
        raise ValueError(f"p0 must have length {tm.N}.")
    if steps < 0:
        raise ValueError("steps must be >= 0.")

    f = np.zeros(steps + 1, dtype=np.float64)

    # t = 0
    if include_t0:
        f[0] = hitter(p)

    # t = 1..steps
    if tm.backend == "sparse":
        Pdot = tm.P.dot
        for t in range(1, steps + 1):
            p = Pdot(p)
            f[t] = hitter(p)
    else:
        P = tm.P
        for t in range(1, steps + 1):
            p = P @ p
            f[t] = hitter(p)

    surv = float(p.sum())
    return f, surv

def stationary_power(
    tm: MarkovTransition,
    tol: float = 1e-12,
    max_iter: int = 200_000,
    lazy: float = 0.0,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    # 幂迭代法计算平稳分布
    if x0 is None:
        p = np.ones(tm.N, dtype=np.float64) / tm.N
    else:
        p = np.asarray(x0, dtype=np.float64).reshape(-1)
        p /= p.sum()

    if lazy != 0.0 and not (0.0 < lazy < 1.0):
        raise ValueError("lazy must be in (0,1)")

    for _ in range(max_iter):
        p_next = tm._step(p)
        if lazy:
            p_next = (1.0 - lazy) * p_next + lazy * p
        s = p_next.sum()
        if s:
            p_next /= s
        if np.linalg.norm(p_next - p, ord=1) < tol:
            break
        p = p_next

    return p

def stationary_eigs(tm: MarkovTransition) -> np.ndarray:
    # 使用特征值分解求平稳分布
    if tm.backend == "dense":
        w, v = np.linalg.eig(tm.P)
        k = int(np.argmin(np.abs(w - 1.0)))
        vec = np.real(v[:, k])
    else:
        w, v = spla.eigs(tm.P, k=1, which="LM")
        vec = np.real(v[:, 0])

    vec = np.maximum(vec, 0.0)
    if vec.sum() == 0:
        vec = np.abs(vec)
    return vec / vec.sum()

def stationary_solve(tm: MarkovTransition) -> np.ndarray:
    # 解线性方程 (I - P)pi = 0 求解平稳分布，并加约束 sum(pi)=1
    if tm.backend == "dense":
        A = np.eye(tm.N) - tm.P
        b = np.zeros(tm.N)
        A[-1, :] = 1.0
        b[-1] = 1.0
        pi = np.linalg.solve(A, b)
    else:
        I = sp.eye(tm.N, format="csc")
        A = (I - tm.P).tolil()
        A[-1, :] = 1.0
        A = A.tocsc()
        b = np.zeros(tm.N)
        b[-1] = 1.0
        pi = spla.spsolve(A, b)

    pi = np.maximum(np.real(pi), 0.0)
    return pi / pi.sum()