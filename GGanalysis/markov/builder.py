from GGanalysis.markov.state_space import *
from GGanalysis.markov.transition import MarkovTransition
import numpy as np
import scipy.sparse as sp
from typing import Any, List, Sequence, Optional, Tuple, Literal, Callable, Iterable, Optional

class TransitionBuilder():
    '''
    转移矩阵构建器（Builder）。
    作用：
    - 收集转移概率（三元组）
    - 支持稀疏 / 稠密两种后端
    - 最终生成一个 Transition 对象

    约定使用“列随机矩阵”（column-stochastic）
    - add(from_id, to_id, p) 的含义是：
        P[to_id, from_id] += p
      即：从 from_id 状态转移到 to_id 的概率为 p
    '''
    def __init__(
        self,
        space: StateSpace,
        backend: Literal["sparse", "dense"] = "sparse",   # "sparse" 或 "dense"
        dtype: Any = np.float64,
    ) -> None:
        if backend not in ("sparse", "dense"):
            raise ValueError("backend must be 'sparse' or 'dense'.")

        self.space = space
        self.backend = backend
        self.dtype = dtype
        self.N = space.N

        if self.backend == "dense":
            # 稠密矩阵：直接分配 NxN 数组
            self._P = np.zeros((self.N, self.N), dtype=self.dtype)
        else:
            # 稀疏矩阵：先用 COO 形式收集 triplets
            self._rows: List[int] = []
            self._cols: List[int] = []
            self._data: List[float] = []

        # build cache / dirty tracking
        self._built_matrix: Optional[MarkovTransition] = None
        self._dirty: bool = True

    def add(self, from_id: int, to_id: int, p: float) -> None:
        '''
        添加一条转移 from_id --(p)--> to_id
        注意：内部矩阵存储的是 P[to_id, from_id] 对应列向量
        '''
        # 输入检查
        if p == 0:
            return
        if not (0 <= from_id < self.N and 0 <= to_id < self.N):
            raise ValueError("from_id/to_id out of bounds.")
        if p < 0:
            raise ValueError("Transition probability p must be >= 0.")
        
        self._dirty = True  # 标记当前进行了修改
        if self.backend == "dense":
            self._P[to_id, from_id] += float(p)
        else:
            self._rows.append(int(to_id))
            self._cols.append(int(from_id))
            self._data.append(float(p))

    def add_state(self, s_from: Sequence[int], s_to: Sequence[int], p: float) -> None:
        '''
        使用 状态向量 而不是 id 添加转移。
        '''
        i = self.space.state_to_id(s_from)
        j = self.space.state_to_id(s_to)
        self.add(i, j, p)

    def add_rule(
        self,
        rule_fn: Callable[[np.ndarray], Iterable[Tuple[Sequence[int], float]]],
        subset: Optional[Selector] = None,
    ) -> None:
        '''
        使用规则函数批量生成转移
        rule_fn:
            输入当前状态 state，返回若干 (next_state, prob)
        subset:
            可选 selector，仅在指定子集上应用规则
        '''
        if subset is None:
            ids = np.arange(self.N, dtype=np.int64)
        else:
            ids = self.space.select_ids(subset)

        for sid in ids:
            s = self.space.id_to_state(int(sid))
            for s2, p in rule_fn(s):
                self.add(int(sid), self.space.state_to_id(s2), float(p))

    def build(self, check=False) -> "MarkovTransition":
        '''完成构建，生成 Transition 对象'''
        if self.backend == "dense":
            return MarkovTransition(self.space, P=self._P, backend="dense")
        rows = np.asarray(self._rows, dtype=np.int64)
        cols = np.asarray(self._cols, dtype=np.int64)
        data = np.asarray(self._data, dtype=self.dtype)
        # COO -> CSR
        P = sp.coo_matrix((data, (rows, cols)), shape=(self.N, self.N)).tocsr()
        P.sum_duplicates()

        M = MarkovTransition(self.space, P=P, backend="sparse")
        if check:
            M.check_and_fix(
                on_underflow="error",
                on_overflow="error",
                fill_unreachable="error",
                warn=True,
            )
        return M
    
    @property
    def matrix(self) -> "MarkovTransition":
        '''构造当前的矩阵'''
        if not self._dirty:
            return self._built_matrix
        tm = self.build()
        self._built_matrix = tm
        self._dirty = False
        return tm