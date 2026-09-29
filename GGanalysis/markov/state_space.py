from typing import Any, List, Sequence, Tuple, Union, Literal
from numbers import Integral
import numpy as np
import scipy.sparse as sp

__all__ = [
    'Selector',
    'StateSpace',
]

# 定义状态选择的输入
SelectorAtom = Union[
    int,              # 固定某个值，例如 第 k 维取 3
    None,             # 全选（等价于 :)
    slice,            # 范围选择，例如 1:5 或 ::2
    Sequence[int],    # 离散集合，例如 [0,2,4]
    np.ndarray        # 离散集合（numpy 版本）
]
Selector = Union[Sequence[SelectorAtom], Tuple[SelectorAtom, ...]]


def _as_1d_int_array(x: Union[Sequence[int], np.ndarray]) -> np.ndarray:
    raw = np.asarray(x)
    arr = raw.astype(np.int64, copy=False).ravel()
    if not np.all(raw.ravel() == arr):
        raise ValueError("Index values must be integers.")
    return arr

class StateSpace():
    '''
    离散多维状态空间(Discrete multi-dimensional state space)
    使用混合进制方式将多维状态编码为一维编号。

    maxes : Sequence[int]
        每一维的最大取值（包含上界），例如：
        [a_max, b_max, c_max, d_max]
        表示第 k 维的合法取值是 0 .. maxes[k]

    oob : Literal["clip", "wrap", "error"]
        越界(out-of-bounds)处理策略
        - "clip"  : 小于 0 的值截断为 0 大于 max 的值截断为 max
        - "wrap"  : 对 (max+1) 取模，形成循环状态空间
        - "error" : 一旦越界立即抛出 ValueError
    '''

    def __init__(
        self,
        maxes: Sequence[int],
        oob: Literal["clip", "wrap", "error"] = "error",
    ) -> None:
        if not all(isinstance(m, Integral) and m >= 0 for m in maxes):
            raise ValueError("All elements in maxes must be non-negative integers.")
        if oob not in ("clip", "wrap", "error"):
            raise ValueError("oob must be one of {'clip', 'wrap', 'error'}.")
        self.maxes: Tuple[int, ...] = tuple(int(m) for m in maxes)
        self.oob: Literal["clip", "wrap", "error"] = oob

        # 状态空间维度数
        self.dims: int = len(self.maxes)
        # 每一维的长度（max + 1）
        self.shape: Tuple[int, ...] = tuple(m + 1 for m in self.maxes)
        # 混合进制编号定义 C-order（行优先）
        # 例如 shape=(A,B,C,D) -> strides=(B*C*D, C*D, D, 1)
        strides = [1] * self.dims
        for i in range(self.dims - 2, -1, -1):
            strides[i] = strides[i + 1] * self.shape[i + 1]
        self.strides: Tuple[int, ...] = tuple(strides)

        # 总状态数 N
        self.N: int = int(np.prod(self.shape, dtype=np.int64))

    @classmethod
    def from_shape(
        cls,
        shape: Sequence[int],
        oob: Literal["clip", "wrap", "error"] = "error",
    ) -> "StateSpace":
        '''根据每一维的长度构造状态空间，比包含上界的 ``maxes`` 更直观。'''
        if not all(isinstance(size, Integral) and size > 0 for size in shape):
            raise ValueError("All elements in shape must be positive integers.")
        return cls([int(size) - 1 for size in shape], oob=oob)

    def max_state(self) -> np.ndarray:
        '''返回最大状态向量 [max_0, ..., max_{d-1}]（便于外部查看/计算）。'''
        return np.asarray(self.maxes, dtype=np.int64)

    def normalize_state(self, state: Sequence[int]) -> np.ndarray:
        '''
        把输入 state 按越界策略规范化，保证返回的状态每维都在合法范围内。
        - clip: 裁剪到 [0,max]
        - wrap: 取模到 [0,max]
        - error: 越界则抛异常
        '''
        if len(state) != self.dims:
            raise ValueError(f"State must have length {self.dims}. Got {len(state)}.")
        raw = np.asarray(state)
        s = raw.astype(np.int64, copy=True)
        if not np.all(raw == s):
            raise ValueError("All state values must be integers.")
        if self.oob == "clip":
            # 裁剪到合法范围
            for k, m in enumerate(self.maxes):
                if s[k] < 0:
                    s[k] = 0
                elif s[k] > m:
                    s[k] = m
        elif self.oob == "wrap":
            # 循环空间：对每维做 mod (max+1)
            for k, m in enumerate(self.maxes):
                mod = m + 1
                s[k] = int(s[k] % mod)
        else:  # error
            # 越界直接报错
            for k, m in enumerate(self.maxes):
                if s[k] < 0 or s[k] > m:
                    raise ValueError(f"State out of bounds at dim {k}: {s[k]} not in [0,{m}].")
        return s

    def state_to_id(self, state: Sequence[int]) -> int:
        '''
        把多维 state 编码成一维 id(0..N-1)
        进行混合进制编码 mixed-radix 类似把多维坐标按 strides 展开
        id = sum(state[k] * strides[k])
        '''
        s = self.normalize_state(state)
        idx = 0
        for v, stride in zip(s, self.strides):
            idx += int(v) * int(stride)
        return int(idx)

    def states_to_ids(self, states: Union[Sequence[Sequence[int]], np.ndarray]) -> np.ndarray:
        '''批量把形状为 ``(数量, 维度数)`` 的状态数组编码为一维编号。'''
        raw = np.asarray(states)
        if raw.ndim != 2 or raw.shape[1] != self.dims:
            raise ValueError(f"states must have shape (n, {self.dims}).")
        values = raw.astype(np.int64, copy=True)
        if not np.all(raw == values):
            raise ValueError("All state values must be integers.")
        maximum = np.asarray(self.maxes, dtype=np.int64)
        if self.oob == "clip":
            np.clip(values, 0, maximum, out=values)
        elif self.oob == "wrap":
            values %= maximum + 1
        elif np.any((values < 0) | (values > maximum)):
            raise ValueError("states contain out-of-bounds values.")
        return values @ np.asarray(self.strides, dtype=np.int64)

    def id_to_state(self, idx: int) -> np.ndarray:
        '''
        把一维 id 解码回多维 state（与 state_to_id 互逆）。
        解码：依次除以 stride 得到每一维的坐标。
        '''
        if idx < 0 or idx >= self.N:
            raise ValueError(f"id must be in [0,{self.N-1}], got {idx}.")
        
        out = np.zeros(self.dims, dtype=np.int64)
        rem = int(idx)
        for k, stride in enumerate(self.strides):
            base = self.shape[k]      # 当前维的进制（长度）
            out[k] = rem // stride    # 当前维的值
            rem = rem % stride        # 剩余部分留给后续维

        return out

    def ids_to_states(self, ids: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        '''批量把一维编号解码为形状为 ``(数量, 维度数)`` 的状态数组。'''
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if np.any((ids < 0) | (ids >= self.N)):
            raise ValueError(f"ids must be in [0, {self.N - 1}].")
        if ids.size == 0:
            return np.empty((0, self.dims), dtype=np.int64)
        return np.column_stack(np.unravel_index(ids, self.shape, order="C"))

    def delta(self, state: Sequence[int], dtype=np.float64) -> np.ndarray:
        '''
        在指定状态上的 delta 分布（概率向量）：
        P(X = state) = 1，其它为 0
        '''
        p = np.zeros(self.N, dtype=dtype)
        p[self.state_to_id(state)] = 1.0
        return p

    def zeros(self, dtype=np.float64) -> np.ndarray:
        p = np.zeros(self.N, dtype=dtype)
        return p

    def _atom_to_indexer(self, atom: SelectorAtom, dim: int) -> Union[int, slice, np.ndarray]:
        '''
        把某一维的 selector 片段(atom)转成 numpy 可以用的索引器。

        selector 始终采用 Python/NumPy 索引语义，不受状态坐标 ``oob`` 设置影响：
        - None -> slice(None)  (全选)
        - int  -> 固定某个值，允许 -1、-2 等合法负索引
        - slice -> 原样保留，由 numpy 解释端点、负数和步长
        - list/ndarray -> 显式值集合，允许合法负索引，并采用集合语义去重

        状态坐标仍由 ``normalize_state`` 按 ``oob`` 处理；selector 与状态坐标是
        两套有意区分的语义。
        '''
        size = self.shape[dim]
        if atom is None:
            return slice(None)
        # Python 整数索引：[-size, size-1] 合法，负数从末尾计数。
        if isinstance(atom, Integral):
            index = int(atom)
            if index < -size or index >= size:
                raise IndexError(
                    f"Selector index {index} is out of bounds for axis {dim} with size {size}."
                )
            return index + size if index < 0 else index
        # slice 始终遵循原生 numpy 切片语义，不应用 clip/wrap/error。
        if isinstance(atom, slice):
            return atom
        # 离散集合逐项使用 Python 整数索引规则；selector 表示区域，因此去除重复值。
        arr = _as_1d_int_array(atom)
        if arr.size == 0:
            return arr
        if np.any((arr < -size) | (arr >= size)):
            raise IndexError(
                f"Selector contains out-of-bounds values for axis {dim} with size {size}."
            )
        arr = np.where(arr < 0, arr + size, arr)
        return np.unique(arr)

    def _view(self, dist: np.ndarray) -> np.ndarray:
        dist = np.asarray(dist)
        if dist.size != self.N:
            raise ValueError(f"Vector length must be {self.N}.")
        return dist.reshape(self.shape)

    def _is_block_selector(self, selector: Selector) -> bool:
        if not isinstance(selector, (list, tuple)) or len(selector) != self.dims:
            return False
        for a in selector:
            if a is None or isinstance(a, (slice, Integral)):
                continue
            return False
        return True
    
    def _make_block_accessor(self, selector: Selector):
        '''
        根据 selector 生成一个访问器：
        返回 (kind, accessor)

        kind:
            "scalar" : 单点
            "block"  : 连续块 view
            "mask"   : 布尔 mask
        accessor:
            - scalar: (dist, idx) -> None
            - block : (dist) -> ndarray view
            - mask  : (dist) -> ndarray view (boolean indexed)
        '''
        if self._is_block_selector(selector):
            # 快路径
            idx = self.parse_selector(selector)
            shape = self.shape
            def access(dist):
                dv = dist.reshape(shape)
                block = dv[idx]
                return block, dv, idx
            return "block", access
        else:
            # 慢路径
            m = self.mask(selector, sparse=False, dtype=bool)
            def access(dist):
                return dist[m]
            return "mask", access

    def hit(self, dist: np.ndarray, selector: Selector) -> float:
        dist = np.asarray(dist)
        if dist.size != self.N:
            raise ValueError(f"Vector length must be {self.N}.")
        kind, access = self._make_block_accessor(selector)
        if kind == "block":
            block, _, _ = access(dist)
            return float(block) if np.isscalar(block) else float(block.sum())
        else:  # mask
            return float(access(dist).sum())

    def clear(self, dist: np.ndarray, selector: Selector) -> None:
        dist = np.asarray(dist)
        if dist.size != self.N:
            raise ValueError(f"Vector length must be {self.N}.")
        kind, access = self._make_block_accessor(selector)
        if kind == "block":
            block, dv, idx = access(dist)
            if np.isscalar(block):
                dv[idx] = 0.0
            else:
                block[...] = 0.0
        else:  # mask
            dist[self.mask(selector)] = 0.0

    def hit_and_clear(self, dist: np.ndarray, selector: Selector) -> float:
        dist = np.asarray(dist)
        if dist.size != self.N:
            raise ValueError(f"Vector length must be {self.N}.")
        kind, access = self._make_block_accessor(selector)
        if kind == "block":
            block, dv, idx = access(dist)
            if np.isscalar(block):
                hit = float(block)
                dv[idx] = 0.0
                return hit
            else:
                hit = float(block.sum())
                block[...] = 0.0
                return hit
        else:  # mask
            m = self.mask(selector)
            hit = float(dist[m].sum())
            dist[m] = 0.0
            return hit

    def compile_hit_and_clear(self, selector: Selector):
        kind, access = self._make_block_accessor(selector)
        if kind == "block":
            def f(dist):
                block, dv, idx = access(dist)
                if np.isscalar(block):
                    hit = float(block)
                    dv[idx] = 0.0
                    return hit
                else:
                    hit = float(block.sum())
                    block[...] = 0.0
                    return hit
            return f
        else:
            m = self.mask(selector)
            def f(dist):
                hit = float(dist[m].sum())
                dist[m] = 0.0
                return hit
            return f

    def parse_selector(self, selector: Selector) -> Tuple[Union[int, slice, np.ndarray], ...]:
        '''
        把传入的 selector 规范化为 per-dim 的索引器 tuple，供 numpy 使用。
        selector 必须是长度 = dims 的 list/tuple。
        '''
        if not isinstance(selector, (list, tuple)):
            raise TypeError("selector must be a list/tuple of per-dimension selectors.")
        if len(selector) != self.dims:
            raise ValueError(f"selector must have length {self.dims}, got {len(selector)}.")
        return tuple(self._atom_to_indexer(selector[d], d) for d in range(self.dims))

    def mask(self, selector: Selector, sparse: bool = False, dtype=bool,
             dense_limit: int = 50_000_000) -> Union[np.ndarray, "sp.csr_matrix"]:
        '''
        生成一维 mask（长度 N），表示哪些状态被 selector 选中。
        - sparse=False：生成 dense mask（np.ndarray，长度 N）
        - sparse=True ：生成稀疏 mask（CSR，形状 (N,1)）

        dense 的实现方式：
        1) 先创建一个 shape=(A,B,C,...) 的全零网格
        2) grid[idx]=1（idx 是 per-dim 的 slice/int/array）
        3) grid.ravel('C') 展平得到一维 mask（与 state_to_id 编码一致）
        '''
        idx = self.parse_selector(selector)

        if sparse:
            # 稀疏 mask：先拿 ids，再构造 (N,1) 的 CSR 列向量
            ids = self.select_ids(selector)
            if sp is None:
                raise RuntimeError("scipy is required for sparse masks.")
            if ids.size == 0:
                return sp.csr_matrix((self.N, 1), dtype=dtype)
            row = ids.astype(np.int64)
            col = np.zeros_like(row)
            data = np.ones_like(row, dtype=dtype)
            return sp.coo_matrix((data, (row, col)), shape=(self.N, 1)).tocsr()

        # 防止 N 太大导致 dense mask 内存爆炸
        if self.N > dense_limit:
            raise MemoryError(
                f"N={self.N} too large for dense mask by default. "
                f"Use sparse=True or select_ids()."
            )

        mask = np.zeros(self.N, dtype=dtype)
        if self._is_block_selector(selector):
            # int/slice/None 不触发多个数组的配对高级索引，保留零额外 ID 数组的快路径。
            mask.reshape(self.shape)[idx] = 1
        else:
            # 离散集合需要笛卡尔积语义；select_ids 已统一实现该语义。
            mask[self.select_ids(selector)] = 1
        return mask

    def select_ids(self, selector: Selector) -> np.ndarray:
        '''
        返回满足 selector 的所有一维编号 ids(np.ndarray)。
        速度较慢，适合调试 / 小规模状态空间

        实现策略：
        1) 把 selector 解析为 per-dim indexer
        2) 每一维 indexer 转成显式取值列表 values[d]
        3) 用 meshgrid + ravel_multi_index 快速生成全部 ids
        '''
        idx = self.parse_selector(selector)
        values: List[np.ndarray] = []
        sizes: List[int] = []

        for d, ix in enumerate(idx):
            if isinstance(ix, int):
                v = np.array([ix], dtype=np.int64)

            elif isinstance(ix, slice):
                # 直接让 numpy 解释负数、越界端点和负步长，确保所有 selector API 一致。
                v = np.arange(self.shape[d], dtype=np.int64)[ix]
            else:
                # ndarray 索引器：已经是显式集合
                v = np.asarray(ix, dtype=np.int64).ravel()  # 转换类型并展平为1维
            values.append(v)
            sizes.append(int(v.size))

        # 有任何一维为空，则整体为空
        if any(s == 0 for s in sizes):
            return np.empty((0,), dtype=np.int64)
        # K = 选中状态数量（笛卡尔积）
        # K = int(np.prod(sizes, dtype=np.int64))
        grids = np.meshgrid(*values, indexing="ij")  # 生成每维坐标网格，得到grids这个list
        multi = tuple(g.reshape(-1) for g in grids)  # 展平成一维坐标列表
        ids = np.ravel_multi_index(multi, dims=self.shape, mode="raise", order="C")
        return ids.astype(np.int64, copy=False)

    def select_slice(self, selector: Selector) -> slice:
        '''
        若 selector 对应的状态在 1D 编码中为连续块，返回 ``slice(start, stop)``。
        不满足连续性条件时抛出 ValueError。

        与 ``select_ids()`` 不同，本方法不分配状态数组，时间复杂度 O(dims)，
        返回的 slice 可直接用于 numpy 数组的快速切片索引。

        连续性条件（C-order 混合进制编码下的充要条件）：
        从第一个取值数量 > 1 的维度开始，之后的所有维度必须是该维度的全选
        （即 ``:`` 或等效的 ``slice(None)``）。在此之前，每个维度只能是单值
        或连续 range(step=1)。

        Examples
        --------
        >>> ss = StateSpace([89, 72])
        >>> ss.select_slice([5, None])        # pity=5 的所有位置
        slice(365, 438, None)
        >>> ss.select_slice([3, slice(10, 20)])  # pity=3, pos 10..19
        slice(229, 239, None)
        >>> ss.select_slice([slice(3, 6), None])  # pity=3..5 的所有位置
        slice(219, 438, None)
        >>> ss.select_slice([None, 5])         # 不连续，抛出 ValueError
        '''
        idx = self.parse_selector(selector)
        values: List[np.ndarray] = []
        sizes: List[int] = []

        for d, ix in enumerate(idx):
            if isinstance(ix, int):
                v = np.array([ix], dtype=np.int64)

            elif isinstance(ix, slice):
                step = 1 if ix.step is None else int(ix.step)
                if step != 1:
                    raise ValueError(
                        f"select_slice requires step=1, got step={step} at dim {d}."
                    )
                v = np.arange(self.shape[d], dtype=np.int64)[ix]
            else:
                # ndarray 索引器
                v = np.asarray(ix, dtype=np.int64).ravel()

            # 检查取值是否连续（step=1）
            if len(v) > 1:
                if not np.all(np.diff(v) == 1):
                    raise ValueError(
                        f"Values at dim {d} are not contiguous (step must be 1)."
                    )

            values.append(v)
            sizes.append(int(v.size))

        # 空结果
        if any(s == 0 for s in sizes):
            return slice(0, 0)

        # 检查连续性条件：首个非单例维度之后，所有维度必须是全选
        found_first_var = False
        for d in range(self.dims):
            if sizes[d] > 1:
                if found_first_var:
                    # 之后出现的多值维度必须是该维度的全选
                    if sizes[d] != self.shape[d] or int(values[d][0]) != 0:
                        raise ValueError(
                            f"Selector is not a contiguous 1D block: "
                            f"dim {d} has {sizes[d]} values but the full "
                            f"dimension size is {self.shape[d]}. "
                            f"After the first varying dimension, all subsequent "
                            f"dimensions must be full-range (:)."
                        )
                else:
                    found_first_var = True
            elif found_first_var:
                # 首个变化维度之后，单例维度必须实际为全选（即 shape[d] == 1）
                if self.shape[d] != 1:
                    raise ValueError(
                        f"Selector is not a contiguous 1D block: "
                        f"dim {d} has only 1 value, but its full size is "
                        f"{self.shape[d]}. After the first varying dimension, "
                        f"all subsequent dimensions must be full-range (:)."
                        f"\nTip: use [{', '.join(':' if j==d else '...' for j in range(self.dims))}] "
                        f"instead of a fixed index here."
                    )

        # 计算起止位置
        # start = sum(values[d][0] * strides[d])
        start = sum(int(v[0]) * s for v, s in zip(values, self.strides))
        total = 1
        for sz in sizes:
            total *= sz
        return slice(start, start + total)
