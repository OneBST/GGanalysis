from GGanalysis.markov.state_space import StateSpace, Selector
from GGanalysis.markov.transition import MarkovTransition, validate_matrix
from copy import deepcopy
import numpy as np
import scipy.sparse as sp
from typing import Any, List, Sequence, Optional, Tuple, Literal, Callable, Iterable

class ProbabilityMatrixBuilder:
    '''
    转移矩阵构建器。

    - 收集转移概率（三元组）
    - 支持稀疏 / 稠密两种后端
    - 生成独立概率矩阵快照，不要求列和为 1

    ``from_space``、``to_space`` 分别定义起点和终点，形状为
    ``(to_space.N, from_space.N)``。``dtype`` 要求实浮点类型。
    起点使用列索引，不要求矩阵为方阵或列和为 1。
    ``add(from_id, to_id, p)`` 将概率累加到 ``P[to_id, from_id]``，
    即从 from_id 状态转移到 to_id 的概率为 p。
    '''
    def __init__(
        self,
        from_space: StateSpace,
        to_space: StateSpace,
        backend: Literal["sparse", "dense"] = "sparse",   # "sparse" 或 "dense"
        dtype: Any = np.float64,
    ) -> None:
        if backend not in ("sparse", "dense"):
            raise ValueError("backend must be 'sparse' or 'dense'.")

        self._from_space = deepcopy(from_space)
        self._to_space = deepcopy(to_space)
        self._backend = backend
        self._dtype = np.dtype(dtype)
        if np.dtype(dtype).kind != "f":
            raise ValueError("dtype must be real floating-point.")

        if self.backend == "dense":
            # 稠密矩阵：直接分配终点数 × 起点数数组
            self._P = np.zeros(self.shape, dtype=self.dtype)
        else:
            # 稀疏矩阵：先用 COO 形式收集 triplets
            self._rows: List[int] = []
            self._cols: List[int] = []
            self._data: List[float] = []

        # build cache / dirty tracking
        self._built_matrix: Optional[np.ndarray | sp.csr_matrix] = None
        self._dirty: bool = True

    @property
    def from_space(self) -> StateSpace:
        return deepcopy(self._from_space)

    @property
    def to_space(self) -> StateSpace:
        return deepcopy(self._to_space)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def N(self) -> int:
        return self._from_space.N

    @property
    def shape(self) -> tuple[int, int]:
        return self._to_space.N, self._from_space.N

    def add(self, from_id: int, to_id: int, p: float) -> None:
        '''
        添加一条转移 from_id --p-> to_id
        注意：内部矩阵存储的是 P[to_id, from_id] 对应列向量
        '''
        self.add_many([from_id], [to_id], [p])

    def add_state(self, s_from: Sequence[int], s_to: Sequence[int], p: float) -> None:
        '''
        使用 状态向量 而不是 id 添加转移。
        '''
        i = self._from_space.state_to_id(s_from)
        j = self._to_space.state_to_id(s_to)
        self.add(i, j, p)

    def add_many(
        self,
        from_ids: Sequence[int] | np.ndarray,
        to_ids: Sequence[int] | np.ndarray,
        probabilities: Sequence[float] | np.ndarray,
    ) -> None:
        '''批量添加编号形式的转移，避免逐边执行 Python 方法调用。'''
        from_ids = self._ids(from_ids, self._from_space.N)
        to_ids = self._ids(to_ids, self._to_space.N)
        probabilities = np.asarray(probabilities, dtype=self.dtype).reshape(-1)
        if not (len(from_ids) == len(to_ids) == len(probabilities)):
            raise ValueError("from_ids, to_ids and probabilities must have equal lengths.")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("Transition probabilities must be finite and >= 0.")
        keep = probabilities != 0
        if not np.any(keep):
            return
        self._dirty = True
        self._built_matrix = None
        from_ids, to_ids, probabilities = from_ids[keep], to_ids[keep], probabilities[keep]
        if self.backend == "dense":
            # np.add.at 能正确累加重复的 (to, from) 坐标。
            np.add.at(self._P, (to_ids, from_ids), probabilities)
        else:
            self._rows.extend(to_ids.tolist())
            self._cols.extend(from_ids.tolist())
            self._data.extend(probabilities.tolist())

    def add_state_many(
        self,
        from_states: Sequence[Sequence[int]] | np.ndarray,
        to_states: Sequence[Sequence[int]] | np.ndarray,
        probabilities: Sequence[float] | np.ndarray,
    ) -> None:
        '''批量编码状态向量并添加转移。'''
        from_ids = self._from_space.states_to_ids(from_states)
        to_ids = self._to_space.states_to_ids(to_states)
        self.add_many(from_ids, to_ids, probabilities)

    def add_rule(
        self,
        rule_fn: Callable[[np.ndarray], Iterable[Tuple[Sequence[int], float]]],
        subset: Optional[Selector] = None,
    ) -> None:
        '''
        使用规则函数批量生成转移。

        Parameters
        ----------
        rule_fn : callable
            输入当前状态 state，返回若干 (next_state, prob)
        subset : optional
            可选 selector，仅在指定子集上应用规则
        '''
        if subset is None:
            ids = np.arange(self.N, dtype=np.int64)
        else:
            ids = self._from_space.select_ids(subset)

        for sid in ids:
            s = self._from_space.id_to_state(int(sid))
            for s2, p in rule_fn(s):
                self.add(int(sid), self._to_space.state_to_id(s2), float(p))

    @staticmethod
    def _ids(values, size: int) -> np.ndarray:
        """拒绝非整数编号，避免静默截断。"""
        array = np.asarray(values).reshape(-1)
        if array.size and (array.dtype.kind not in "iu" or
                           np.any(array < 0) or np.any(array >= size)):
            raise ValueError("state IDs must be integers within the state space.")
        return array.astype(np.int64)

    def build(self) -> np.ndarray | sp.csr_matrix:
        """返回独立快照；内部缓存不会暴露给调用者。"""
        if self._dirty:
            if self.backend == "dense":
                matrix = self._P.copy()
            else:
                matrix = sp.coo_matrix(
                    (self._data, (self._rows, self._cols)),
                    shape=self.shape, dtype=self.dtype,
                ).tocsr()
                matrix.sum_duplicates()
            validate_matrix(matrix, self.shape, "probability matrix")
            self._built_matrix = matrix
            self._dirty = False
        return self._built_matrix.copy()

    @property
    def matrix(self) -> np.ndarray | sp.csr_matrix:
        """当前矩阵的独立快照。"""
        return self.build()


class TransitionBuilder:
    """同一状态空间上的构建器，返回独立 ``MarkovTransition``。

    ``space`` 是状态空间，``backend`` 可选 sparse/dense，``dtype`` 为实浮点类型。
    ``build(check=True)`` 只验证概率守恒，不修复概率。
    """

    def __init__(self, space: StateSpace,
                 backend: Literal["sparse", "dense"] = "sparse",
                 dtype: Any = np.float64) -> None:
        self._builder = ProbabilityMatrixBuilder(space, space, backend, dtype)

    @property
    def space(self) -> StateSpace:
        return self._builder.from_space

    @property
    def N(self) -> int:
        return self._builder.N

    @property
    def backend(self) -> str:
        return self._builder.backend

    @property
    def dtype(self) -> np.dtype:
        return self._builder.dtype

    def add(self, from_id: int, to_id: int, p: float) -> None:
        """添加一条概率边。"""
        self._builder.add(from_id, to_id, p)

    def add_many(self, from_ids, to_ids, probabilities) -> None:
        """批量添加概率边，重复坐标累加。"""
        self._builder.add_many(from_ids, to_ids, probabilities)

    def add_state(self, s_from: Sequence[int], s_to: Sequence[int], p: float) -> None:
        """通过状态向量添加概率边。"""
        self._builder.add_state(s_from, s_to, p)

    def add_state_many(self, from_states, to_states, probabilities) -> None:
        """批量添加状态向量形式的边。"""
        self._builder.add_state_many(from_states, to_states, probabilities)

    def add_rule(self, rule_fn: Callable, subset: Optional[Selector] = None) -> None:
        """在起点子集上应用转移规则。"""
        self._builder.add_rule(rule_fn, subset)

    def build(self, check: bool = False) -> MarkovTransition:
        """返回独立快照，可选检查每列概率和为 1。"""
        result = MarkovTransition(self.space, self._builder.build(), self.backend)
        if check:
            result.validate()
        return result

    @property
    def matrix(self) -> MarkovTransition:
        """返回独立快照，修改结果不影响后续构建。"""
        return self.build()


__all__ = ["ProbabilityMatrixBuilder", "TransitionBuilder"]
