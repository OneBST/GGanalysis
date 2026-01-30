from GGanalysis.markov.state_space import *
import numpy as np
import scipy.sparse as sp
import warnings
from typing import Any, List, Union, Literal

class MarkovTransition():
    '''
    马尔科夫链转移矩阵
    使用 CSR 格式以加速 P @ p 计算。

    矩阵 P 的含义：
        P[to, from] = Pr(X_{t+1} = to | X_t = from)

    支持两种后端：
    - dense : numpy.ndarray (N, N)
    - sparse: scipy.sparse 矩阵 (推荐使用 CSR 便于列操作和 SpMV)

    TODO 考虑实用性后决定是否加入 自动从普通转移矩阵生成吸收链的功能：给定原始转移矩阵 P 和一组吸收态，自动构造一个新矩阵。也可以据此构造一个计算吸收概率和吸收时间期望的函数。
    '''
    def __init__(
        self,
        space: StateSpace,
        P: Union[np.ndarray, sp.spmatrix],  # np.ndarray 或 scipy.sparse 矩阵
        backend: Literal["sparse", "dense"] = "sparse",  # "sparse" 或 "dense"
    ) -> None:
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

    def __call__(self, p: np.ndarray) -> np.ndarray:
        '''Sugar for one-step propagation: tm(p) == tm.step(p).'''
        return self._step(p)

    def __matmul__(self, other: Any) -> Any:
        '''支持使用 `tm @ p` 表示一步状态传播 p_next = P @ p

        仅当 other 是长度为 N 的一维向量时才视为状态向量，
        否则不支持矩阵与矩阵的乘法（避免语义混淆）
        '''
        arr = np.asarray(other)
        if arr.ndim == 1 and arr.size == self.N:
            return self._step(arr)
        raise TypeError("TransitionMatrix @ only supports vector multiplication.")
    
    def copy(self) -> "MarkovTransition":
        '''Shallow copy of TransitionMatrix with a copied underlying matrix.'''
        if self.backend == "dense":
            P2 = np.array(self.P, copy=True)
        else:
            P2 = self.P.copy()
        return MarkovTransition(self.space, P=P2, backend=self.backend)

    # 完成矩阵转移
    def _step(self, p: np.ndarray) -> np.ndarray:
        # 单步转移 p_next = P @ p
        p = np.asarray(p).reshape(-1)
        if p.size != self.N:
            raise ValueError(f"p must have length {self.N}.")
        return self.P @ p

    def check_and_fix(
        self,
        tol: float = 1e-12,  # 容忍误差限
        on_underflow: Literal["self_loop", "renorm", "error"] = "self_loop",
        on_overflow: Literal["error", "renorm"] = "error",
        fill_unreachable: Literal["error", "self_loop"] = "self_loop",
        warn: bool = True,
    ) -> None:
        '''
        概率校验和修复 要求每一列概率和应为1

        - 列和 == 0
            视为该状态无转移定义（不可达/未定义）
            默认在对角线上补一个自环概率 1
        - 列和 < 1 - tol
            概率缺失
            - self_loop 把缺失概率补到自环
            - renorm    整列按比例放大
            - error     直接报错
        - 列和 > 1 + tol
            概率溢出
            - error   直接报错
            - renorm  整列按比例缩放
        '''
        # 输入检查
        if on_underflow not in ("self_loop", "renorm", "error"):
            raise ValueError("on_underflow must be self_loop|renorm|error")
        if on_overflow not in ("error", "renorm"):
            raise ValueError("on_overflow must be error|renorm")
        if fill_unreachable not in ("self_loop", "error"):
            raise ValueError("fill_unreachable must be self_loop|error")
        col_sum = self._column_sums()
        # 记录需要补到对角线（自环）的概率
        diag_add = np.zeros(self.N, dtype=np.float64)

        for j, s in enumerate(col_sum):
            if abs(s - 1.0) <= tol:
                continue
            # 列和为 0：完全没有定义转移
            if s == 0.0:
                if fill_unreachable == "error":
                    raise ValueError(f"Column {j} has no outgoing probability (sum=0).")
                diag_add[j] += 1.0
                if warn:
                    warnings.warn(f"Column {j} state {self.space.id_to_state(j)} sum=0; filled self-loop with 1.0", RuntimeWarning)
                continue
            # 列和小于 1：概率缺失
            if s < 1.0 - tol:
                if on_underflow == "error":
                    raise ValueError(f"Column {j} state {self.space.id_to_state(j)} sums to {s} < 1 (missing probability).")
                if on_underflow == "renorm":
                    self._scale_column(j, 1.0 / s)
                    if warn:
                        warnings.warn(f"Column {j} state {self.space.id_to_state(j)} renormalized by factor {1.0/s}", RuntimeWarning)
                else:
                    diag_add[j] += (1.0 - s)  # 默认：把缺失概率补到自环
                    if warn:
                        warnings.warn(f"Column {j} state {self.space.id_to_state(j)} sums to {s}; added {1.0-s} to self-loop", RuntimeWarning)
            # 列和大于 1：概率溢出
            elif s > 1.0 + tol:
                if on_overflow == "error":
                    raise ValueError(f"Column {j} state {self.space.id_to_state(j)} sums to {s} > 1 (probability overflow).")
                else:
                    self._scale_column(j, 1.0 / s)
                    if warn:
                        warnings.warn(f"Column {j} state {self.space.id_to_state(j)} overflow {s}; renormalized", RuntimeWarning)
        # 统一补加自环
        if np.any(diag_add != 0):
            self._add_to_diagonal(diag_add)
        # 最终一致性检查
        col_sum2 = self._column_sums()
        if np.max(np.abs(col_sum2 - 1.0)) > 1e-8 and warn:
            warnings.warn(f"After check_and_fix, max|col_sum-1|={np.max(np.abs(col_sum2-1.0))}.", RuntimeWarning)

    def _column_sums(self) -> np.ndarray:
        # 返回每一列的概率和（长度 N 的一维数组）。
        if self.backend == "dense":
            return self.P.sum(axis=0, dtype=np.float64)
        else:
            # 稀疏矩阵 sum(axis=0) 返回 1xN 矩阵，这里拉平
            return np.asarray(self.P.sum(axis=0)).ravel()
        
    def _add_to_diagonal(self, diag_add: np.ndarray) -> None:
        # 用来补到自环的函数
        if self.backend == "dense":
            self.P[np.arange(self.N), np.arange(self.N)] += diag_add.astype(self.P.dtype, copy=False)
        else:
            D = sp.diags(diag_add, offsets=0, shape=(self.N, self.N), format="csr")
            self.P = (self.P + D).tocsr()

    def _scale_column(self, col: int, factor: float) -> None:
        # 将某一列整体按 factor 缩放
        if self.backend == "dense":
            self.P[:, col] *= factor
        else:
            start, end = self.P.indptr[col], self.P.indptr[col + 1]
            if start != end:
                self.P.data[start:end] *= factor