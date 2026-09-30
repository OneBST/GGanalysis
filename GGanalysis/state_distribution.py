"""带有限内部状态的花费分布。

本模块可以看作 :class:`FiniteDist` 的有限状态推广，核心约定如下：

1. ``FiniteDist.dist[t]`` 只记录“花费为 t”的概率，是标量系数多项式。
2. ``StateDist.coeff[t, s]`` 记录“累计花费为 t、当前状态为 s”的联合概率，
   是向量系数多项式。它表示已经给定初始条件后的概率分布。
3. ``StateKernel.coeff[t, p, k]`` 记录“从状态 k 出发，花费 t 后到达状态 p”
   的联合条件概率，是矩阵系数多项式。其中第 1 维是结束状态，第 2 维是开始状态。
4. 全部对象都把花费轴放在第 0 轴。花费相加对应第 0 轴上的卷积，状态衔接对应
   对中间状态求和。
5. 使用列向量约定，所以 ``kernel_b @ kernel_a`` 表示先执行 ``kernel_a``，
   再执行 ``kernel_b``；``kernel @ state_dist`` 表示核作用于已有联合分布。
6. ``kernel ** n`` 表示同一个过程连续执行 n 次。``StateDist`` 没有自卷积运算，
   因为它没有独立的开始状态轴，不能表达阶段之间的状态衔接。

直接算法在花费轴上枚举拆分，并对每组系数执行矩阵乘法。
FFT 算法沿花费轴做实数 FFT，然后在每个频率点执行矩阵乘法。
"""

from __future__ import annotations

from numbers import Real
from typing import Iterable, Literal, Union

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from GGanalysis.distribution_1d import FiniteDist


ConvolutionMethod = Literal["auto", "direct", "fft"]
ArrayLike = Union[list, np.ndarray]


def _trim_cost_axis(coeff: np.ndarray) -> np.ndarray:
    """删除末尾全为 0 的花费层，并保证至少保留一个花费层。"""
    nonzero = np.flatnonzero(np.any(coeff != 0, axis=tuple(range(1, coeff.ndim))))
    return coeff[: nonzero[-1] + 1] if nonzero.size else coeff[:1]


def _readonly_view(array: np.ndarray) -> np.ndarray:
    """返回不可写视图，避免外部修改数组后破坏对象内部一致性。"""
    view = array.view()
    view.flags.writeable = False
    return view


def _check_method(method: str) -> ConvolutionMethod:
    if method not in {"auto", "direct", "fft"}:
        raise ValueError("method 必须为 'auto'、'direct' 或 'fft'")
    return method  # type: ignore[return-value]


def _choose_method(cost_a: int, cost_b: int, state_size: int) -> ConvolutionMethod:
    """使用保守的经验规则选择后端；调用者始终可以显式指定计算方法。"""
    if cost_a * cost_b <= 16:
        return "direct"
    if max(cost_a, cost_b) <= 8:
        return "direct" if state_size < 128 else "fft"
    return "fft"


class StateDist:
    r"""非负整数花费与有限状态的联合分布。

    ``coeff[t, s]`` 表示累计花费为 ``t`` 且当前状态为 ``s`` 的概率质量。
    本类允许概率和不为 1 的非负测度，以便通过普通加法和数乘构造加权混合。

    参数
    ----
    coeff : list、numpy.ndarray 或 StateDist
        形状必须为 ``(花费长度, 状态数)``。
    trim_tail_zeros : bool
        是否删除末尾所有状态系数都严格为 0 的花费层，默认删除。

    注意
    ----
    ``StateDist`` 表示已经应用初始条件后的联合分布，不是状态转移过程。
    因此两个 ``StateDist`` 之间只定义加权叠加，不定义卷积。
    需要重复执行一个过程时，应使用 ``StateKernel`` 进行组合或乘方。
    """

    def __init__(self, coeff: Union[ArrayLike, "StateDist"], trim_tail_zeros: bool = True) -> None:
        source = coeff._coeff if isinstance(coeff, StateDist) else coeff
        array = np.array(source, dtype=np.float64, copy=True)
        if array.ndim != 2:
            raise ValueError("StateDist 的 coeff 必须具有 (花费, 状态) 形状")
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("StateDist 的花费轴和状态轴不能为空")
        self._coeff = _trim_cost_axis(array) if trim_tail_zeros else array

    @property
    def coeff(self) -> np.ndarray:
        """联合分布系数的不可写视图，形状为 ``(花费, 状态)``。"""
        return _readonly_view(self._coeff)

    @property
    def cost_size(self) -> int:
        """花费轴长度。"""
        return self._coeff.shape[0]

    @property
    def state_size(self) -> int:
        """有限状态数量。"""
        return self._coeff.shape[1]

    def marginal_cost(self, *, tail_mass: float | None = None) -> FiniteDist:
        """对状态求和，可用 ``tail_mass`` 显式传入已知未存储质量。

        不从系数推断尾质量，也不据此补全期望或方差。
        """
        return FiniteDist(self._coeff.sum(axis=1), tail_mass=tail_mass)

    def marginal_state(self) -> FiniteDist:
        """对所有花费求和，将状态编号视为非负整数并返回 ``FiniteDist``。"""
        return FiniteDist(self._coeff.sum(axis=0))

    def total_mass(self) -> float:
        """返回所有花费和状态上的总概率质量。"""
        return float(self._coeff.sum())

    def normalized(self) -> "StateDist":
        """返回总概率质量归一化为 1 的新对象。"""
        mass = self.total_mass()
        if mass == 0:
            raise ZeroDivisionError("不能归一化总概率质量为 0 的 StateDist")
        return StateDist(self._coeff / mass)

    def condition_state(self, state: int) -> FiniteDist:
        """返回给定当前状态时，花费的条件分布。"""
        column = self._coeff[:, state]
        mass = float(column.sum())
        if mass == 0:
            raise ZeroDivisionError("用于条件化的状态事件概率为 0")
        return FiniteDist(column / mass)

    def condition_cost(self, cost: int) -> np.ndarray:
        """返回给定累计花费时，当前状态的条件概率向量。"""
        row = self._coeff[cost].copy()
        mass = float(row.sum())
        if mass == 0:
            raise ZeroDivisionError("用于条件化的花费事件概率为 0")
        return row / mass

    def trim_tail_zeros(self) -> "StateDist":
        """返回删除末尾全 0 花费层后的新对象。"""
        return StateDist(self._coeff)

    @classmethod
    def delta(cls, state: int, state_size: int, cost: int = 0) -> "StateDist":
        """构造全部概率集中在指定花费和状态上的 delta 联合分布。"""
        if state_size <= 0 or cost < 0 or not 0 <= state < state_size:
            raise ValueError("state_size、state 或 cost 不合法")
        coeff = np.zeros((cost + 1, state_size))
        coeff[cost, state] = 1.0
        return cls(coeff)

    @classmethod
    def from_product(cls, cost_dist: Union[FiniteDist, ArrayLike], state_prob: ArrayLike) -> "StateDist":
        """由相互独立的花费分布和状态分布构造联合分布。"""
        cost = cost_dist.dist if isinstance(cost_dist, FiniteDist) else np.asarray(cost_dist)
        state = np.asarray(state_prob)
        if cost.ndim != 1 or state.ndim != 1:
            raise ValueError("cost_dist 和 state_prob 必须都是一维数组")
        return cls(np.multiply.outer(cost, state))

    @classmethod
    def mixture(cls, dists: Iterable["StateDist"], weights: Iterable[float]) -> "StateDist":
        """按给定权重线性混合多个状态联合分布。"""
        dist_list, weight_list = list(dists), list(weights)
        if len(dist_list) != len(weight_list):
            raise ValueError("dists 和 weights 的长度必须相同")
        pairs = list(zip(dist_list, weight_list))
        if not pairs:
            raise ValueError("混合时至少需要一个分布")
        result = pairs[0][1] * pairs[0][0]
        for dist, weight in pairs[1:]:
            result = result + weight * dist
        return result

    def __add__(self, other: "StateDist") -> "StateDist":
        """按花费 0 对齐，将两个联合概率测度逐项相加。"""
        if not isinstance(other, StateDist):
            return NotImplemented
        if self.state_size != other.state_size:
            raise ValueError("两个 StateDist 的状态数不匹配")
        length = max(self.cost_size, other.cost_size)
        coeff = np.zeros((length, self.state_size))
        coeff[: self.cost_size] += self._coeff
        coeff[: other.cost_size] += other._coeff
        return StateDist(coeff)

    def __mul__(self, scalar: Real) -> "StateDist":
        """返回联合概率测度与数值的乘积。"""
        if not isinstance(scalar, Real):
            return NotImplemented
        return StateDist(self._coeff * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Real) -> "StateDist":
        """返回联合概率测度除以数值的结果。"""
        return StateDist(self._coeff / scalar)

    def __getitem__(self, item):
        """使用 numpy 索引方式读取系数，并返回副本。"""
        return self._coeff[item].copy()

    def __len__(self) -> int:
        """返回花费轴长度。"""
        return self.cost_size

    def __repr__(self) -> str:
        return f"StateDist(花费长度={self.cost_size}, 状态数={self.state_size}, 总质量={self.total_mass():.12g})"


class StateKernel:
    r"""同时记录花费和有限状态转移的概率核。

    ``coeff[t, p, k]`` 表示从开始状态 ``k`` 出发，花费 ``t`` 后到达结束状态
    ``p`` 的联合条件概率。对任意固定的开始状态 ``k``，合法概率核应满足：

    .. math::

        \sum_{t,p} K_t[p,k] = 1.

    核允许为长方形，即输入状态空间和输出状态空间可以不同。只要前一个核的输出
    状态数与后一个核的输入状态数相同，就可以进行组合。

    运算约定
    --------
    - ``kernel_b @ kernel_a``：先执行 ``kernel_a``，再执行 ``kernel_b``。
    - ``kernel @ state_dist``：将状态转移核作用于已有联合分布。
    - ``kernel ** n``：同一个方形核连续执行 ``n`` 次。
    - ``a + b``、``weight * a``：构造核的线性组合或随机混合。

    参数
    ----
    coeff : list、numpy.ndarray 或 StateKernel
        形状必须为 ``(花费长度, 输出状态数, 输入状态数)``。
    trim_tail_zeros : bool
        是否删除末尾所有矩阵元素都严格为 0 的花费层，默认删除。
    """

    def __init__(self, coeff: Union[ArrayLike, "StateKernel"], trim_tail_zeros: bool = True) -> None:
        source = coeff._coeff if isinstance(coeff, StateKernel) else coeff
        array = np.array(source, dtype=np.float64, copy=True)
        if array.ndim != 3:
            raise ValueError("StateKernel 的 coeff 必须具有 (花费, 结束状态, 开始状态) 形状")
        if any(size == 0 for size in array.shape):
            raise ValueError("StateKernel 的各个轴不能为空")
        self._coeff = _trim_cost_axis(array) if trim_tail_zeros else array

    @property
    def coeff(self) -> np.ndarray:
        """状态转移核系数的不可写视图，轴顺序为花费、结束状态、开始状态。"""
        return _readonly_view(self._coeff)

    @property
    def cost_size(self) -> int:
        """花费轴长度。"""
        return self._coeff.shape[0]

    @property
    def output_state_size(self) -> int:
        """输出（结束）状态数量。"""
        return self._coeff.shape[1]

    @property
    def input_state_size(self) -> int:
        """输入（开始）状态数量。"""
        return self._coeff.shape[2]

    @property
    def is_square(self) -> bool:
        """输入状态数和输出状态数是否相同。只有方形核可以重复执行。"""
        return self.output_state_size == self.input_state_size

    @classmethod
    def identity(cls, state_size: int) -> "StateKernel":
        """构造花费为 0、状态保持不变的单位核。"""
        if state_size <= 0:
            raise ValueError("state_size 必须为正整数")
        return cls(np.eye(state_size)[None, :, :])

    @classmethod
    def from_finite_dist(cls, dist: Union[FiniteDist, ArrayLike], state_size: int) -> "StateKernel":
        r"""将普通花费分布嵌入为不改变状态的核 ``K_t = dist[t] I``。"""
        cost = dist.dist if isinstance(dist, FiniteDist) else np.asarray(dist, dtype=np.float64)
        if cost.ndim != 1:
            raise ValueError("dist 必须为一维数组")
        return cls(cost[:, None, None] * np.eye(state_size)[None, :, :])

    @classmethod
    def deterministic(cls, transition: ArrayLike, cost: int = 0) -> "StateKernel":
        """构造仅在指定花费层具有给定转移矩阵的确定花费核。"""
        matrix = np.asarray(transition, dtype=np.float64)
        if matrix.ndim != 2 or cost < 0:
            raise ValueError("transition 必须为矩阵，且 cost 必须为非负整数")
        coeff = np.zeros((cost + 1,) + matrix.shape)
        coeff[cost] = matrix
        return cls(coeff)

    @classmethod
    def mixture(cls, kernels: Iterable["StateKernel"], weights: Iterable[float]) -> "StateKernel":
        """按给定权重线性混合多个形状相同的状态转移核。"""
        kernel_list, weight_list = list(kernels), list(weights)
        if len(kernel_list) != len(weight_list):
            raise ValueError("kernels 和 weights 的长度必须相同")
        pairs = list(zip(kernel_list, weight_list))
        if not pairs:
            raise ValueError("混合时至少需要一个状态转移核")
        result = pairs[0][1] * pairs[0][0]
        for kernel, weight in pairs[1:]:
            result = result + weight * kernel
        return result

    def marginal_transition(self) -> np.ndarray:
        """对花费求和，返回普通状态转移矩阵 ``[结束状态, 开始状态]``。"""
        return self._coeff.sum(axis=0)

    def cost_dist(self, start_state: int) -> FiniteDist:
        """给定开始状态，对结束状态求和，返回该阶段的花费分布。"""
        return FiniteDist(self._coeff[:, :, start_state].sum(axis=1))

    def validate(self, atol: float = 1e-12) -> None:
        """检查负概率及每个开始状态对应的总概率质量是否为 1。"""
        minimum = float(self._coeff.min())
        if minimum < -atol:
            raise ValueError(f"状态转移核包含负系数 {minimum}")
        masses = self._coeff.sum(axis=(0, 1))
        if not np.allclose(masses, 1.0, rtol=0.0, atol=atol):
            raise ValueError("每个开始状态对应的所有花费和结束状态概率之和必须为 1")

    def _resolved_method(self, other_cost_size: int, state_size: int, method: str) -> ConvolutionMethod:
        """检查用户指定的方法，并在 auto 模式下调用经验规则。"""
        checked = _check_method(method)
        return _choose_method(self.cost_size, other_cost_size, state_size) if checked == "auto" else checked

    def compose(self, previous: "StateKernel", method: ConvolutionMethod = "auto") -> "StateKernel":
        r"""组合两个核，返回先执行 ``previous``、再执行 ``self`` 的结果。

        系数满足：

        .. math::

            C_t[p,k] = \sum_{\tau,j} self_\tau[p,j]
            previous_{t-\tau}[j,k].

        ``method`` 可以为 ``"direct"``、``"fft"`` 或 ``"auto"``。
        """
        if not isinstance(previous, StateKernel):
            raise TypeError("previous 必须为 StateKernel")
        if previous.output_state_size != self.input_state_size:
            raise ValueError("previous 的输出状态数与当前核的输入状态数不匹配")
        selected = self._resolved_method(previous.cost_size, self.input_state_size, method)
        coeff = self._compose_direct(previous) if selected == "direct" else self._compose_fft(previous)
        return StateKernel(coeff)

    def _compose_direct(self, previous: "StateKernel") -> np.ndarray:
        """在花费轴上直接卷积，并用批量矩阵乘法收缩中间状态。

        选择花费长度较短的一侧保留 Python 循环，另一侧交给 numpy 的批量
        ``matmul``。与逐花费层双循环完全等价，但能显著减少 Python 调用开销。
        """
        current = self._coeff
        previous_coeff = previous._coeff
        output = np.zeros(
            (
                len(current) + len(previous_coeff) - 1,
                self.output_state_size,
                previous.input_state_size,
            ),
            dtype=np.result_type(current, previous_coeff),
        )

        if len(current) <= len(previous_coeff):
            for current_cost, matrix in enumerate(current):
                output[current_cost:current_cost + len(previous_coeff)] += (
                    matrix[None, :, :] @ previous_coeff
                )
        else:
            for previous_cost, matrix in enumerate(previous_coeff):
                output[previous_cost:previous_cost + len(current)] += (
                    current @ matrix[None, :, :]
                )
        return output

    def _compose_fft(self, previous: "StateKernel") -> np.ndarray:
        """沿花费轴做 FFT，在每个频率点执行批量矩阵乘法。"""
        output_len = self.cost_size + previous.cost_size - 1
        fft_len = next_fast_len(output_len)
        left = rfft(self._coeff, n=fft_len, axis=0)
        right = rfft(previous._coeff, n=fft_len, axis=0)
        return irfft(left @ right, n=fft_len, axis=0)[:output_len]

    def apply(self, dist: StateDist, method: ConvolutionMethod = "auto") -> StateDist:
        """将本核作用于状态联合分布，返回传播后的联合分布。"""
        if not isinstance(dist, StateDist):
            raise TypeError("dist 必须为 StateDist")
        if dist.state_size != self.input_state_size:
            raise ValueError("StateDist 的状态数与核的输入状态数不匹配")
        selected = self._resolved_method(dist.cost_size, self.input_state_size, method)
        coeff = self._apply_direct(dist) if selected == "direct" else self._apply_fft(dist)
        return StateDist(coeff)

    def _apply_direct(self, dist: StateDist) -> np.ndarray:
        """使用直接花费卷积和矩阵向量乘法传播联合分布。"""
        output = np.zeros((self.cost_size + dist.cost_size - 1, self.output_state_size))
        for tk in range(self.cost_size):
            for td in range(dist.cost_size):
                output[tk + td] += self._coeff[tk] @ dist._coeff[td]
        return output

    def _apply_fft(self, dist: StateDist) -> np.ndarray:
        """沿花费轴做 FFT，在每个频率点执行矩阵向量乘法。"""
        output_len = self.cost_size + dist.cost_size - 1
        fft_len = next_fast_len(output_len)
        kernel_hat = rfft(self._coeff, n=fft_len, axis=0)
        dist_hat = rfft(dist._coeff, n=fft_len, axis=0)
        return irfft(np.einsum("fpk,fk->fp", kernel_hat, dist_hat, optimize=True),
                     n=fft_len, axis=0)[:output_len]

    def apply_power(self, n: int, dist: StateDist, method: ConvolutionMethod = "auto") -> StateDist:
        """使用二进制快速幂思想，将本核连续作用 ``n`` 次。

        与 ``(kernel ** n) @ dist`` 数学等价，但只需要最终联合分布时，此方法可以
        避免构造最后的完整核，通常更节省内存。
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("n 必须为非负整数")
        if not self.is_square:
            raise ValueError("只有输入、输出状态空间相同的方形核才能重复执行")
        if dist.state_size != self.input_state_size:
            raise ValueError("StateDist 的状态数与核的输入状态数不匹配")
        result = dist
        base = self
        power = n
        while power:
            if power & 1:
                result = base.apply(result, method=method)
            power >>= 1
            if power:
                base = base.compose(base, method=method)
        return result

    def __matmul__(self, other):
        """使用 ``@`` 组合状态转移核，或将核作用于 StateDist。"""
        if isinstance(other, StateKernel):
            return self.compose(other)
        if isinstance(other, StateDist):
            return self.apply(other)
        return NotImplemented

    def __pow__(self, n: int) -> "StateKernel":
        """使用二进制快速幂返回本方形核连续执行 ``n`` 次的组合核。"""
        if not isinstance(n, int) or n < 0:
            raise ValueError("n 必须为非负整数")
        if not self.is_square:
            raise ValueError("只有输入、输出状态空间相同的方形核才能进行乘方")
        result = StateKernel.identity(self.input_state_size)
        base = self
        power = n
        while power:
            if power & 1:
                result = base.compose(result)
            power >>= 1
            if power:
                base = base.compose(base)
        return result

    def __add__(self, other: "StateKernel") -> "StateKernel":
        """按花费 0 对齐，将两个核的系数逐项相加。"""
        if not isinstance(other, StateKernel):
            return NotImplemented
        if self._coeff.shape[1:] != other._coeff.shape[1:]:
            raise ValueError("两个核的输入或输出状态数不匹配")
        length = max(self.cost_size, other.cost_size)
        coeff = np.zeros((length,) + self._coeff.shape[1:])
        coeff[: self.cost_size] += self._coeff
        coeff[: other.cost_size] += other._coeff
        return StateKernel(coeff)

    def __mul__(self, scalar: Real) -> "StateKernel":
        """返回状态转移核与数值的乘积。"""
        if not isinstance(scalar, Real):
            return NotImplemented
        return StateKernel(self._coeff * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Real) -> "StateKernel":
        """返回状态转移核除以数值的结果。"""
        return StateKernel(self._coeff / scalar)

    def __getitem__(self, item):
        """使用 numpy 索引方式读取系数，并返回副本。"""
        return self._coeff[item].copy()

    def __len__(self) -> int:
        """返回花费轴长度。"""
        return self.cost_size

    def __repr__(self) -> str:
        return (f"StateKernel(花费长度={self.cost_size}, 输出状态数={self.output_state_size}, "
                f"输入状态数={self.input_state_size})")


__all__ = ["StateDist", "StateKernel"]
