from GGanalysis.markov.state_space import StateSpace
from GGanalysis.state_distribution import StateKernel
import numpy as np
import scipy.sparse as sp
from typing import Any, Union, Literal, Optional, Sequence

class MarkovTransition():
    '''
    马尔科夫链转移矩阵
    使用 CSR 格式以加速 P @ p 计算。

    矩阵 P 的含义：
        P[to, from] = Pr(X_{t+1} = to | X_t = from)

    支持两种后端：
    - dense : numpy.ndarray (N, N)
    - sparse: scipy.sparse 矩阵（内部统一为 CSR，以便高效执行 P @ p）

    TODO 考虑实用性后决定是否加入 自动从普通转移矩阵生成吸收链的功能：给定原始转移矩阵 P 和一组吸收态，自动构造一个新矩阵。也可以据此构造一个计算吸收概率和吸收时间期望的函数。
    '''
    def __init__(
        self,
        space: StateSpace,
        P: Union[np.ndarray, sp.spmatrix],  # np.ndarray 或 scipy.sparse 矩阵
        backend: Optional[Literal["sparse", "dense"]] = None,
    ) -> None:
        if backend is None:
            backend = "sparse" if sp.issparse(P) else "dense"
        # 参数合法性检查
        if backend not in ("sparse", "dense"):
            raise ValueError("backend must be 'sparse' or 'dense'.")

        self.space: StateSpace = space
        self.P: Union[np.ndarray, sp.spmatrix] = P
        self.backend = backend
        self.N = self.space.N

        if self.backend == "dense":
            if not isinstance(self.P, np.ndarray):
                raise TypeError("Dense backend requires numpy.ndarray.")
            if self.P.shape != (self.N, self.N):
                raise ValueError("P has wrong shape.")
        else:
            if not sp.issparse(self.P):
                raise TypeError("Sparse backend requires scipy.sparse matrix.")
            if self.P.shape != (self.N, self.N):
                raise ValueError("P has wrong shape.")
            # 统一转换为 CSR 格式
            if not isinstance(self.P, sp.csr_matrix):
                self.P = self.P.tocsr()
        validate_matrix(self.P, (self.N, self.N), "P")

    def __call__(self, p: np.ndarray) -> np.ndarray:
        '''Sugar for one-step propagation: tm(p) == tm.step(p).'''
        return self.step(p)

    def __matmul__(self, other: Any) -> Any:
        '''支持使用 `tm @ p` 表示一步状态传播 p_next = P @ p

        仅当 other 是长度为 N 的一维向量时才视为状态向量，
        否则不支持矩阵与矩阵的乘法（避免语义混淆）
        '''
        arr = np.asarray(other)
        if arr.ndim == 1 and arr.size == self.N:
            return self.step(arr)
        raise TypeError("TransitionMatrix @ only supports vector multiplication.")
    
    def copy(self) -> "MarkovTransition":
        '''Shallow copy of TransitionMatrix with a copied underlying matrix.'''
        if self.backend == "dense":
            P2 = np.array(self.P, copy=True)
        else:
            P2 = self.P.copy()
        return MarkovTransition(self.space, P=P2, backend=self.backend)

    def step(self, p: np.ndarray) -> np.ndarray:
        '''执行一次状态传播 ``p_next = P @ p``。'''
        return self._step(p)

    # 完成矩阵转移（保留内部入口以兼容现有分析代码）
    def _step(self, p: np.ndarray) -> np.ndarray:
        # 单步转移 p_next = P @ p
        p = np.asarray(p)
        if p.ndim != 1 or p.size != self.N:
            raise ValueError(f"p must be a one-dimensional vector of length {self.N}.")
        return self.P @ p

    def validate(self, atol: float = 1e-12) -> None:
        """检查形状、有限非负系数与列和为 1，不修改矩阵。

        ``atol`` 为列和的绝对误差容限，不用于修复概率缺失。
        """
        validate_matrix(self.P, (self.N, self.N), "P")
        validate_mass(np.asarray(self.P.sum(axis=0)).ravel(), 1.0, atol)


def validate_probability_vector(
    vector: Sequence[float] | np.ndarray, size: int, name: str = "initial",
    *, normalized: bool = False, flatten: bool = False, atol: float = 1e-12,
) -> np.ndarray:
    """返回有限非负状态质量向量的副本，不自动归一化。

    默认要求一维且长度为 size；flatten=True 接受可展平的输入。
    normalized=True 时通过 validate_mass 检查总质量为 1，atol 为其容差。
    此辅助函数供马尔可夫模块内部使用，不限制非归一化中间质量。
    """
    values = np.array(vector, dtype=float, copy=True)
    if flatten:
        values = values.reshape(-1)
    if values.shape != (size,):
        raise ValueError(f"{name} must be a vector of length {size}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities.")
    if normalized:
        validate_mass(values.sum(), 1.0, atol)
    return values


def validate_matrix(matrix, shape: tuple[int, int], name: str) -> None:
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    data = matrix.data if sp.issparse(matrix) else np.asarray(matrix)
    if not np.all(np.isfinite(data)) or np.any(data < 0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities.")


def validate_mass(actual, expected, atol: float = 1e-12) -> None:
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative.")
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        raise ValueError("Probability mass is not conserved.")


def validate_steps(value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("max_steps must be a nonnegative integer.")


def prepare_event_matrices(
    matrices: Sequence[np.ndarray | sp.spmatrix | Sequence[Sequence[float]]] | StateKernel,
) -> tuple[sp.csr_matrix, ...]:
    """将单步事件转移整理为独立 CSR 矩阵，并检查联合概率守恒。（转换单抽核为稀疏格式）

    第 r 个矩阵记录本步计 r 次事件及状态转移的联合概率。
    各矩阵为同大小方阵，其和须列归一化；单个矩阵不要求列和为 1。
    StateKernel 的累计轴在此解释为事件数量；不修复或归一化输入。
    """
    if isinstance(matrices, StateKernel):
        matrices = matrices.coeff
    blocks = tuple(sp.csr_matrix(matrix, dtype=float, copy=True) for matrix in matrices)
    if not blocks:
        raise ValueError("event_matrices must not be empty.")
    size = blocks[0].shape[0]
    for block in blocks:
        validate_matrix(block, (size, size), "event matrix")
        block.eliminate_zeros()
    validate_mass(np.asarray(sum(blocks).sum(axis=0)).ravel(), 1)
    return blocks
