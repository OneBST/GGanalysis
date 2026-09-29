from __future__ import annotations

from typing import Iterable, Union
import numpy as np
from scipy.signal import convolve
from scipy.fft import irfft, next_fast_len, rfft
from collections import OrderedDict


def _irfft_probability(
    spectrum: np.ndarray,
    size: int,
    *,
    atol: float = 1e-14,
    rtol: float = 1e-12,
) -> np.ndarray:
    '''将 ``rfft`` 频域值安全地变回实数概率系数。

    ``irfft`` 直接返回实数。这里只额外检查 DC/Nyquist 边界、有限值和
    非负性；容差内的微小负值视为舍入误差并裁剪为 0，明显负值则报错。
    '''
    spectrum = np.asarray(spectrum)
    if size <= 0 or spectrum.ndim != 1:
        raise ValueError('size must be positive and spectrum must be one-dimensional.')
    expected_spectrum_size = size // 2 + 1
    if len(spectrum) != expected_spectrum_size:
        raise ValueError(
            f'spectrum length must be {expected_spectrum_size} for real output size {size}.'
        )
    if not np.all(np.isfinite(spectrum)):
        raise FloatingPointError('FFT spectrum contains NaN or infinity.')

    spectrum_scale = max(1.0, float(np.max(np.abs(spectrum))))
    spectrum_tolerance = float(atol + rtol * spectrum_scale)
    boundary_imag = [abs(float(spectrum[0].imag))]
    if size % 2 == 0:
        boundary_imag.append(abs(float(spectrum[-1].imag)))
    max_boundary_imag = max(boundary_imag)
    if max_boundary_imag > spectrum_tolerance:
        raise FloatingPointError(
            f'rFFT boundary has imaginary residual {max_boundary_imag:.3e} '
            f'(tolerance {spectrum_tolerance:.3e}).'
        )

    values = np.asarray(irfft(spectrum, n=size), dtype=float)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError('inverse rFFT contains NaN or infinity.')
    value_scale = max(1.0, float(np.max(np.abs(values))))
    value_tolerance = float(atol + rtol * value_scale)
    minimum = float(np.min(values))
    if minimum < -value_tolerance:
        raise FloatingPointError(
            f'inverse rFFT produced a negative probability {minimum:.3e} '
            f'(tolerance {value_tolerance:.3e}).'
        )

    result = values.copy()
    result[result < 0.0] = 0.0
    return result


def _fft_probability_converged(
    values: np.ndarray,
    previous: np.ndarray | None,
    expected_exp: float,
    expected_mass: float,
    tolerance: float,
) -> tuple[bool, tuple[float, float, float]]:
    '''同时检查期望、质量和相邻 FFT 长度公共前缀是否收敛。'''
    expectation = float(np.dot(np.arange(len(values)), values))
    expectation_error = abs(expectation - expected_exp) / max(1.0, abs(expected_exp))
    mass_error = abs(float(np.sum(values)) - expected_mass) / max(1.0, abs(expected_mass))
    prefix_error = float('inf')
    if previous is not None:
        prefix_error = float(np.sum(np.abs(previous - values[:len(previous)])))
    errors = expectation_error, mass_error, prefix_error
    return previous is not None and max(errors) <= tolerance, errors

def linear_p_increase(base_p=0.01, pity_begin=100, step=1, hard_pity=100):
    '''
    计算线性递增模型的保底参数

    - ``base_p`` : 基础概率
    - ``pity_begin`` ：概率开始上升位置
    - ``step`` ：每次上升概率
    - ``hard_pity`` ：硬保底位置
    '''
    ans = np.zeros(hard_pity+1)
    ans[1:pity_begin] = base_p
    ans[pity_begin:hard_pity+1] = np.arange(1, hard_pity-pity_begin+2) * step + base_p
    return np.minimum(ans, 1)

def calc_expectation(dist: Union['FiniteDist', list, np.ndarray]) -> float:
    '''
    计算离散分布列的期望
    '''
    if isinstance(dist, FiniteDist):
        dist = dist.dist
    else:
        dist = np.asarray(dist)
    x = np.arange(len(dist))
    return float(np.dot(x, dist))

def calc_variance(dist: Union['FiniteDist', list, np.ndarray]) -> float:
    '''
    计算离散分布列的方差
    '''
    if isinstance(dist, FiniteDist):
        dist = dist.dist
    else:
        dist = np.asarray(dist)
    x = np.arange(dist.size, dtype=np.float64)
    ex = float(np.dot(x, dist))
    ex2 = float(np.dot(x * x, dist))
    var = ex2 - ex * ex
    # 数值误差下可能出现 -1e-16 这种，钳到 0
    return float(var) if var > 0.0 else 0.0

def dist_squeeze(dist: Union['FiniteDist', np.ndarray], squeeze_factor) -> 'FiniteDist':
    '''
    按照 squeeze_factor 对分布进行倍数压缩，将压缩部分和存在一起
    '''
    if squeeze_factor <= 0:
        raise ValueError("k must be >= 1")

    # 取底层 1D 数组视图（不修改它）
    arr = dist.dist if isinstance(dist, FiniteDist) else np.asarray(dist)
    arr = np.asarray(arr, dtype=np.float64).ravel()  # 可能拷贝，但不会改原对象

    if arr.size == 0:
        return FiniteDist([1.0])  # 或者按你的语义返回 FiniteDist([0])，看你项目约定

    tail = arr[1:]
    n = tail.size
    m_full = n // squeeze_factor
    rem = n - m_full * squeeze_factor

    out_len = 1 + m_full + (1 if rem else 0)
    out = np.empty(out_len, dtype=np.float64)
    out[0] = float(arr[0])

    # 完整块部分 reshape + sum
    if m_full:
        out[1:1 + m_full] = tail[:m_full * squeeze_factor].reshape(m_full, squeeze_factor).sum(axis=1)
    # 不足块部分单独求和
    if rem:
        out[-1] = float(tail[m_full * squeeze_factor:].sum())
    return FiniteDist(out)

def dist2cdf(dist: Union[np.ndarray, 'FiniteDist']) -> np.ndarray:
    '''
    将分布转为cdf
    '''
    if isinstance(dist, FiniteDist):
        return np.cumsum(dist.dist)
    return np.cumsum(dist)

def cdf2dist(cdf: np.ndarray) -> 'FiniteDist':
    '''
    将cdf转化为分布
    '''
    return FiniteDist.from_cdf(cdf)

def p2dist(pity_p: Union[list, np.ndarray]) -> 'FiniteDist':
    '''
    将保底概率参数转化为分布列
    '''
    return FiniteDist.from_pity_p(pity_p)

def dist2p(dist: Union[np.ndarray, 'FiniteDist']) -> np.ndarray:
    '''
    将分布转换为条件概率表
    '''
    if isinstance(dist, FiniteDist):
        dist = dist.dist
    dist = np.array(dist)
    left_p = np.cumsum(dist[::-1])[::-1]
    return np.divide(dist, left_p, where=left_p!=0, out=np.zeros_like(dist))

def p2exp(pity_p: Union[list, np.ndarray]) -> float:
    '''
    对于列表，认为是概率提升表，返回对应分布期望
    '''
    return calc_expectation(p2dist(pity_p))

def p2var(pity_p: Union[list, np.ndarray]) -> float:
    '''
    对于列表，认为是概率提升表，返回对应分布方差
    '''
    return calc_variance(p2dist(pity_p))

def pad_zero(dist:np.ndarray, target_len) -> np.ndarray:
    '''
    给 numpy 数组末尾补零至指定长度
    '''
    if target_len <= len(dist):
        return dist
    return np.pad(dist, (0, target_len-len(dist)), 'constant', constant_values=0)

def prob_a_greater_than_b(a: Union['FiniteDist', np.ndarray], b: Union['FiniteDist', np.ndarray]) -> float:
    '''计算分布采样时a>b的概率'''
    max_len = max(len(a), len(b))
    b = pad_zero(b[:], max_len)
    a = pad_zero(a[:], max_len)
    prob = np.sum(a[1:] * np.cumsum(b)[:-1])
    return prob

def cut_dist(dist: Union[np.ndarray, 'FiniteDist'], cut_pos) -> np.ndarray:
    '''
    切除分布并重新进行概率归一化，默认切除头部
    '''
    # cut_pos为0则没有进行切除
    if cut_pos == 0:
        return dist
    # 进行了切除后进行归一化
    ans = dist[cut_pos:].copy()
    ans[0] = 0
    return ans/sum(ans)

def calc_item_num_dist(dist_list: list['FiniteDist'], pull) -> 'FiniteDist':
    '''
    根据的获得 0-k 个道具所需抽数分布列表计算使用 pull 抽时获得道具数量分布（第k个位置表达的是≥k的概率）
    此方法会忽略概率太低的长尾部分，并将其概率累加到 k 个道具位置
    '''
    item_num = len(dist_list) - 1
    ans = np.zeros(item_num+1)
    for i in range(0, item_num+1):
        if len(dist_list[i]) <= pull+1:
            ans[i] = 1
        else:
            ans[i] = dist_list[i].cdf[pull]
    ans[0:-1] -= ans[1:].copy()
    return FiniteDist(ans)

def independent_item_num_dist(f_dist: 'FiniteDist', pull: int, c_dist: 'FiniteDist'=None, multi_dist=False) -> Union[list, 'FiniteDist']:
    '''
    **获得道具数量快速计算**

    输入抽取道具所需抽数分布，返回当投入 pull 抽时获得道具数量的分布
    注意，本函数要求输入道具满足每次获取时消耗抽数独立同分布，同时每抽最多获得一个道具

    - ``f_dist`` : 获取道具所需抽数的完整分布
    - ``pull`` : 投入抽数
    - ``c_dist`` : 获取第一个道具所需抽数的条件分布（可选）
    - ``multi_dist`` : 是否以列表形式返回从投入0抽到pull抽的所有分布
    '''
    if c_dist is None:
        c_dist = f_dist
    conv_dist = FiniteDist(f_dist[:pull+1])
    check_dist = FiniteDist(c_dist[:pull+1])
    if multi_dist:
        ans = np.zeros((pull+1, pull+1))  # [获得道具数量, 投入抽数]
        ans[0, :] = 1
        if pull >= 1:
            ans[1, :len(check_dist)] = check_dist.cdf[:]
            ans[1, len(check_dist):] = 1
        for item_num in range(2, pull+1):
            # 每做一次卷积后都要把长度限制在需求长度内以加速计算
            check_dist = FiniteDist((check_dist * conv_dist)[:pull+1])
            if len(check_dist) == 1 and check_dist[0] == 0:  # 由于精度限制，计算几乎截止
                ans[item_num:, :] = 0
                break
            else:
                ans[item_num, :len(check_dist)] = check_dist.cdf[:]
                ans[item_num, len(check_dist):] = 1
        ans[:-1, :] -= ans[1:, :].copy()
        return [FiniteDist(ans[:i+1, i], trim_tail_zeros=False) for i in range(pull+1)]
    else:
        ans = np.zeros(pull+1)
        ans[0] = 1
        if pull >= 1:
            ans[1] = 1 if len(c_dist) <= pull else check_dist.cdf[pull]
        for item_num in range(2, pull+1):
            # 每做一次卷积后都要把长度限制在需求长度内以加速计算
            check_dist = FiniteDist((check_dist * conv_dist)[:pull+1], trim_tail_zeros=False)
            if len(check_dist) == 1 and check_dist[0] == 0:  # 由于精度限制，计算几乎截止
                ans[item_num:] = 0
                break
            else:
                ans[item_num] = 1 if len(check_dist) <= pull else check_dist.cdf[pull]
        ans[:-1] -= ans[1:].copy()
        return FiniteDist(ans, trim_tail_zeros=False)

def calc_bernoulli_obtain(dist: 'FiniteDist', p: float, e_error = 1e-8, max_dist_len=1e5) -> 'FiniteDist':
    '''
    输入基本道具所需抽数分布，计算在每次获得基本道具时以固定概率p成功获得另一种道具的情况下，获得另一种道具所需抽数的概率分布
    和 gacha_layers 中定义的 BernoulliLayer 功能类似

    - ``dist`` : 输入道具分布
    - ``p`` : 获取另一种道具概率
    - ``e_error`` : 容许期望误差比例
    - ``max_dist_len`` : 最长不警告序列长度
    '''
    # 概率为 1 等价于什么也没干
    if p == 1:
        return dist
    exp = 1/p
    def calc_2nd_moment(E, D):
        return (p * D + (2 - p) * E ** 2)/(p ** 2)
    ans_E = dist.exp * exp  # 叠加后的期望
    ans_D = calc_2nd_moment(dist.exp, dist.var) - ans_E ** 2  # 叠加后的方差
    test_len = max(int(ans_E + 10 * ans_D ** 0.5), len(dist), 2)
    previous = None
    while True:
        # 通过实数 FFT 求概率生成函数，并逐次扩大长度检查循环混叠是否收敛。
        fft_len = next_fast_len(test_len, real=True)
        F_dist = rfft(dist.dist, n=fft_len)
        output_spectrum = (p * F_dist) / (1 - (1 - p) * F_dist)
        output_array = _irfft_probability(output_spectrum, fft_len)
        # 解决输出位置0处不是0的问题
        output_array[0] = 0.0
        output_dist = FiniteDist(output_array)
        converged, errors = _fft_probability_converged(
            output_dist.dist,
            previous,
            ans_E,
            float(output_spectrum[0].real),
            e_error,
        )
        if converged or fft_len > max_dist_len:
            if fft_len > max_dist_len and not converged:
                print(
                    'Warning: distribution is too long! len:', fft_len,
                    'Expectation/Mass/Prefix errors:', errors,
                )
            output_dist.exp = ans_E
            output_dist.var = ans_D
            return output_dist
        previous = output_dist.dist.copy()
        test_len = fft_len * 2

def accurate_conv(dist_a: 'FiniteDist', dist_b: 'FiniteDist') -> 'FiniteDist':
    '''有更高数值精确性的卷积，适用于计算分布概率低于1e-18的情况'''
    ans = FiniteDist(convolve(dist_a.dist, dist_b.dist, method='direct'))
    ans.exp = dist_a.exp + dist_b.exp
    ans.var = dist_a.var + dist_b.var
    return ans

class FiniteDist:  # 随机事件为有限个数的分布
    r'''
    **有限长一维分布**

    用于快速进行有限长一维离散分布的相关计算，创建时通过传入分布数组进行初始化，可以代表一个随机变量。
    本质上是对一个 ``numpy`` 数组的封装，通过利用 FFT、快速幂思想、局部缓存等方法使得各类分布相关的运算能方便高效的进行，有良好的算法复杂度。

    - *** 运算** 定义 ``FiniteDist`` 类型和数值之间的 ``*`` 运算为数值乘，将返回 ``FiniteDist.dist`` 和数值进行数乘后的结果；定义两个 ``FiniteDist`` 类型之间的 ``*`` 运算为卷积，即计算两个随机变量相加的分布。
    - **/ 运算** 定义 ``FiniteDist`` 类型和数值之间的 ``/`` 运算为数值乘，将返回 ``FiniteDist.dist`` 和数值进行数乘后的结果； 定义两个 ``FiniteDist`` 类型之间的 ``+`` 运算为分布叠加，将返回将两个 ``FiniteDist.dist`` 加和后的结果。
    - **\*\* 运算** 定义 ``FiniteDist`` 类型与整数的 ``**`` 运算为自卷积，将返回卷积自身数值次后的结果；``FiniteDist`` 变量A与另一个 ``FiniteDist`` 变量B的 ``**`` 运算为 :math:`\sum_{i=0}^{len(B)}{A^B[i]}`。

    **类初始化**

    - ``dist`` : 采用列表、numpy数组或者FiniteDist记录的分布信息进行初始化
    - ``trim_tail_zeros`` : 是否去除分布末尾的0，默认清除。（当尾概率很低又想靠分布长度来体现最极端情况位置时，可以设置为不清除）
    - ``tail_mass`` : 已知但没有存入数组的概率质量；无法确定时为 ``None``

    **类属性**

    - ``dist`` : 用列表、numpy数组、FiniteDist表示在自然数位置的分布。若不传入初始化分布默认为全部概率集中在0位置，即离散卷积运算的幺元。
    - ``exp`` ：这个一维分布的期望
    - ``var`` ：这个一维分布的方差
    - ``p_sum`` ：这个一维分布所有位置的概率的和
    - ``tail_mass`` ：已知但没有存入数组的概率质量，与 ``p_sum`` 分开记录
    - ``entropy_rate`` ：这个一维分布的熵率，即分布的熵除其期望，意义为平均每次尝试的信息量
    - ``randomness_rate`` ：此处定义的随机度为此分布熵率和概率为 :math:`\\frac{1}{exp}`. 的伯努利信源的熵率的比值，越低则说明信息量越低
    
    .. attention:: 
    
        ``FiniteDist.dist`` 返回不可写视图，避免绕过缓存管理直接修改数组。
        局部修改请使用 ``dist[index] = value``；整体替换请使用 ``set_dist``。
        两种入口都会立即清除旧的卷积、统计量、CDF 和近似信息缓存。
    
    **用例**

    .. code:: Python

        # 定义随机变量 dist_a
        dist_a = FiniteDist([0.5, 0.5])
        # 定义随机变量 dist_b
        dist_b = FiniteDist([0, 1])
        # 通过卷积计算两个随机变量叠加 dist_a + dist_b 的分布
        dist_a * dist_b
        # 计算独立同分布的随机变量 dist_b 累加 10 次后的分布
        dist_b ** 10
    '''
    _cache_size: int = 128  # 默认缓存临时分布数量上限
    def __init__(
        self,
        dist: Union[list, np.ndarray, 'FiniteDist', None] = None,
        trim_tail_zeros: bool = True,
        *,
        exp: float | None = None,
        var: float | None = None,
        cdf: Union[list, np.ndarray, None] = None,
        tail_mass: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.__dist = np.ones(1, dtype=float)
        self._pow_cache = OrderedDict()
        self._exp = None
        self._var = None
        self._p_sum = None
        self._cdf = None
        self._entropy_rate = None
        self._randomness_rate = None
        self._tail_mass = None
        self._metadata = None
        self.set_dist(
            [1.0] if dist is None else dist,
            trim_tail_zeros=trim_tail_zeros,
            exp=exp,
            var=var,
            cdf=cdf,
            tail_mass=tail_mass,
            metadata=metadata,
        )

    @classmethod
    def delta(cls, value: int) -> 'FiniteDist':
        '''构造概率全部集中在指定非负整数位置的分布。

        例如 ``FiniteDist.delta(0)`` 是卷积运算的单位元，
        ``FiniteDist.delta(10)`` 表示随机变量必定取值 10。
        '''
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError('value must be a non-negative integer.')
        dist = np.zeros(int(value) + 1, dtype=float)
        dist[int(value)] = 1.0
        return cls(dist)

    @classmethod
    def from_cdf(cls, cdf: Union[list, np.ndarray]) -> 'FiniteDist':
        '''根据累积分布函数构造有限分布。

        输入第 ``i`` 项表示随机变量不大于 ``i`` 的概率。此方法保留原
        ``cdf2dist`` 的行为：长度为 1 时返回概率集中在 0 位置的必然分布。
        '''
        cdf = np.asarray(cdf, dtype=float)
        if cdf.ndim != 1:
            raise ValueError('cdf must be a 1D array.')
        if len(cdf) == 0:
            raise ValueError('cdf must not be empty.')
        if len(cdf) == 1:
            return cls([1])
        dist = cdf.copy()
        dist[1:] -= dist[:-1].copy()
        last_cdf = float(cdf[-1])
        tail_mass = max(0.0, 1.0 - last_cdf) if -1e-12 <= last_cdf <= 1.0 + 1e-12 else None
        return cls(dist, cdf=cdf, tail_mass=tail_mass)

    @classmethod
    def from_pity_p(cls, pity_p: Union[list, np.ndarray]) -> 'FiniteDist':
        '''根据逐次条件成功概率表构造首次成功位置分布。

        ``pity_p[i]`` 表示此前均未成功时，在第 ``i`` 次成功的条件概率。
        按 GGanalysis 的保底表约定，位置 0 通常应为 0，计算从位置 1 开始。
        '''
        pity_p = np.asarray(pity_p, dtype=float)
        if pity_p.ndim != 1:
            raise ValueError('pity_p must be a 1D array.')
        if len(pity_p) == 0:
            raise ValueError('pity_p must not be empty.')
        left_p = 1.0
        dist = np.zeros(len(pity_p), dtype=float)
        for i in range(1, len(pity_p)):
            dist[i] = left_p * pity_p[i]
            left_p *= 1 - pity_p[i]
        return cls(dist, tail_mass=max(0.0, float(left_p)))

    @classmethod
    def mixture(
        cls,
        dists: Iterable['FiniteDist'],
        weights: Iterable[float],
    ) -> 'FiniteDist':
        '''返回多个有限分布的加权线性组合。

        本方法不强制权重非负或权重和为 1，与 ``FiniteDist`` 的加法和数乘
        语义一致。若权重非负且和为 1，结果就是通常意义下的概率混合分布。
        '''
        dist_list = list(dists)
        weight_list = list(weights)
        if len(dist_list) != len(weight_list):
            raise ValueError('dists and weights must have equal lengths.')
        if not dist_list:
            raise ValueError('mixture requires at least one distribution.')
        if not all(isinstance(dist, FiniteDist) for dist in dist_list):
            raise TypeError('all elements in dists must be FiniteDist.')

        target_len = max(len(dist) for dist in dist_list)
        ans = np.zeros(target_len, dtype=float)
        for dist, weight in zip(dist_list, weight_list):
            ans[:len(dist)] += float(weight) * dist.dist
        return cls(ans)

    @staticmethod
    def _coerce_dist(
        dist: Union[list, np.ndarray, 'FiniteDist'],
        trim_tail_zeros: bool,
    ) -> np.ndarray:
        '''复制并检查一维系数数组，保证不与调用者共享可写内存。'''
        source = dist.__dist if isinstance(dist, FiniteDist) else dist
        array = np.array(source, dtype=float, copy=True)
        if array.ndim != 1:
            raise ValueError('dist must be a 1D array.')
        if trim_tail_zeros:
            array = np.trim_zeros(array, 'b')
        if len(array) == 0:
            array = np.zeros(1, dtype=float)
        return array

    @staticmethod
    def _coerce_cdf(cdf: Union[list, np.ndarray, None]) -> np.ndarray | None:
        if cdf is None:
            return None
        array = np.array(cdf, dtype=float, copy=True)
        if array.ndim != 1 or len(array) == 0:
            raise ValueError('cdf must be a non-empty 1D array.')
        return array

    @staticmethod
    def _coerce_tail_mass(tail_mass: float | None) -> float | None:
        if tail_mass is None:
            return None
        value = float(tail_mass)
        if not np.isfinite(value) or value < 0.0 or value > 1.0 + 1e-12:
            raise ValueError('tail_mass must be a finite probability in [0, 1].')
        return value

    def _touch(self) -> None:
        '''分布系数变化后立即清除所有依赖旧系数的缓存和元数据。'''
        self._pow_cache.clear()
        self._exp = None
        self._var = None
        self._p_sum = None
        self._cdf = None
        self._entropy_rate = None
        self._randomness_rate = None
        self._tail_mass = None
        self._metadata = None

    def set_dist(
        self,
        dist: Union[list, np.ndarray, 'FiniteDist'],
        *,
        trim_tail_zeros: bool = True,
        exp: float | None = None,
        var: float | None = None,
        cdf: Union[list, np.ndarray, None] = None,
        tail_mass: float | None = None,
        metadata: dict | None = None,
    ) -> 'FiniteDist':
        '''整体替换系数，并原子设置与新系数匹配的可选信息。

        新数组会先完成复制和形状检查；检查失败时对象保持原状。替换成功
        后，旧的卷积幂、统计量、CDF、尾质量和 metadata 全部失效，再写入
        本次显式提供的新信息。
        '''
        array = self._coerce_dist(dist, trim_tail_zeros)
        prepared_cdf = self._coerce_cdf(cdf)
        prepared_tail_mass = self._coerce_tail_mass(tail_mass)
        prepared_metadata = None if metadata is None else dict(metadata)
        prepared_exp = None if exp is None else float(exp)
        prepared_var = None if var is None else float(var)

        self.__dist = array
        self._touch()
        self._exp = prepared_exp
        self._var = prepared_var
        self._cdf = prepared_cdf
        self._tail_mass = prepared_tail_mass
        self._metadata = prepared_metadata
        return self

    @property
    def dist(self):
        # 防止变量被意外修改导致缓存策略出错
        view = self.__dist.view()
        view.flags.writeable = False
        return view

    @property
    def exp(self) -> float:
        '''分布的期望；第一次访问时计算并缓存。'''
        if self._exp is None:
            self.calc_dist_attribution()
        return self._exp

    @exp.setter
    def exp(self, value: float | None) -> None:
        self._exp = None if value is None else float(value)
        self._entropy_rate = None
        self._randomness_rate = None

    @property
    def var(self) -> float:
        '''分布的方差；第一次访问时计算并缓存。'''
        if self._var is None:
            self.calc_dist_attribution()
        return self._var

    @var.setter
    def var(self, value: float | None) -> None:
        self._var = None if value is None else float(value)

    @property
    def p_sum(self) -> float:
        '''分布的总概率质量；第一次访问时计算并缓存。'''
        if self._p_sum is None:
            self.calc_dist_attribution()
        return self._p_sum

    @property
    def cdf(self) -> np.ndarray:
        '''累积分布函数的不可写视图；第一次访问时计算并缓存。'''
        if self._cdf is None:
            self.calc_cdf()
        view = self._cdf.view()
        view.flags.writeable = False
        return view

    @cdf.setter
    def cdf(self, value: Union[list, np.ndarray, None]) -> None:
        self._cdf = self._coerce_cdf(value)

    @property
    def tail_mass(self) -> float | None:
        '''已知但没有存入 ``dist`` 数组的概率质量；未知时为 ``None``。'''
        return self._tail_mass

    @tail_mass.setter
    def tail_mass(self, value: float | None) -> None:
        self._tail_mass = self._coerce_tail_mass(value)

    @property
    def metadata(self) -> dict | None:
        '''返回近似算法和来源等附加信息的浅拷贝。'''
        return None if self._metadata is None else dict(self._metadata)

    @metadata.setter
    def metadata(self, value: dict | None) -> None:
        self._metadata = None if value is None else dict(value)

    @property
    def entropy_rate(self) -> float:
        '''分布的熵率；第一次访问时计算并缓存。'''
        if self._entropy_rate is None:
            self.calc_entropy_attribution()
        return self._entropy_rate

    @property
    def randomness_rate(self) -> float:
        '''分布相对于同期望伯努利信源的随机度；第一次访问时计算并缓存。'''
        if self._randomness_rate is None:
            self.calc_entropy_attribution()
        return self._randomness_rate
    
    def __iter__(self): 
        return iter(self.__dist)

    def __setitem__(self, sliced, value: Union[int, float, np.ndarray]) -> None:
        '''原地修改一个位置或切片，并立即清除所有依赖旧系数的信息。'''
        self.__dist[sliced] = value
        self._touch()

    def __getitem__(self, sliced):
        '''将numpy切片的方法应用于 ``dist`` 直接取得numpy数组切片'''
        return self.__dist[sliced].copy()

    def trim_tail_zeros(self):
        '''去除分布末尾的0'''
        return FiniteDist(self.dist)

    def validate(self, atol: float = 1e-12, *, require_normalized: bool = True) -> None:
        '''验证有限值、非负性、概率质量及已缓存 CDF 的一致性。

        若 ``tail_mass`` 已知，归一化检查使用 ``p_sum + tail_mass``；否则只
        检查数组内的质量。对于有意表示一般线性序列的对象，可设置
        ``require_normalized=False``，但有限值和非负性仍会检查。
        '''
        if atol < 0:
            raise ValueError('atol must be non-negative.')
        if not np.all(np.isfinite(self.__dist)):
            raise ValueError('distribution contains non-finite coefficients.')
        minimum = float(np.min(self.__dist))
        if minimum < -atol:
            raise ValueError(f'distribution contains a negative coefficient {minimum}.')

        represented_mass = self.p_sum
        accounted_mass = represented_mass + (self._tail_mass or 0.0)
        if require_normalized and not np.isclose(accounted_mass, 1.0, rtol=0.0, atol=atol):
            raise ValueError(
                f'probability mass is {represented_mass} with tail_mass '
                f'{self._tail_mass}; accounted total is {accounted_mass}, not 1.'
            )

        if self._cdf is not None:
            if not np.all(np.isfinite(self._cdf)):
                raise ValueError('cdf contains non-finite values.')
            if np.any(np.diff(self._cdf) < -atol):
                raise ValueError('cdf must be non-decreasing.')
            if float(np.min(self._cdf)) < -atol or float(np.max(self._cdf)) > 1.0 + atol:
                raise ValueError('cdf values must lie in [0, 1].')
            if len(self._cdf) < len(self.__dist):
                raise ValueError('cached cdf is shorter than the distribution.')
            expected = np.cumsum(self.__dist)
            if not np.allclose(self._cdf[:len(expected)], expected, rtol=0.0, atol=atol):
                raise ValueError('cached cdf is inconsistent with distribution coefficients.')
            if len(self._cdf) > len(expected) and not np.allclose(
                self._cdf[len(expected):], expected[-1], rtol=0.0, atol=atol
            ):
                raise ValueError('cached cdf tail is inconsistent with the distribution.')

    def calc_cdf(self):
        '''将自身分布转为cdf返回'''
        self._cdf = dist2cdf(self.dist)
        self._cdf.flags.writeable = False

    def calc_dist_attribution(self, p_error=1e-6) -> None:
        '''
        计算分布的基本属性 ``exp`` ``var`` ``p_sum``
        
        - ``p_error`` ： 容许 ``p_sum`` 和 1 之间的差距，默认为 1e-6
        '''
        self._p_sum = float(sum(self.dist))
        if abs(self._p_sum-1) > p_error:  # 高于阈值认为概率和不为1
            if self._exp is None:
                self._exp = float('nan')
            if self._var is None:
                self._var = float('nan')
            return
        use_pulls = np.arange(self.__len__())
        stored_exp = float(sum(use_pulls * self.dist))
        stored_var = float(sum((use_pulls-stored_exp) ** 2 * self.dist))
        if self._exp is None:
            self._exp = stored_exp
        if self._var is None:
            self._var = stored_var

    def calc_entropy_attribution(self, p_error=1e-6) -> None:
        '''计算分布熵相关属性 ``entropy_rate`` ``randomness_rate``

        - ``p_error`` ： 容许 ``p_sum`` 和 1 之间的差距，默认为 1e-6
        '''
        if abs(self.p_sum-1) > p_error:  # 概率和不为1
            self._entropy_rate = float('nan')
            self._randomness_rate = float('nan')
            return
        # 避免0的对数
        temp = np.zeros(len(self.dist))
        temp[0] = 1
        self._entropy_rate = float(-sum(self.dist * np.log2(self.dist+temp)) / self.exp)
        self._randomness_rate = float(self._entropy_rate / (-1/self.exp * np.log2(1/self.exp) - (1-1/self.exp) * np.log2(1-1/self.exp)))

    def quantile_point(self, quantile_p):
        '''返回分位点位置'''
        return np.searchsorted(self.cdf, quantile_p, side='left')

    def normalized(self) -> 'FiniteDist':
        '''返回分布进行归一化后的结果'''
        mass = float(sum(self.__dist))
        if mass == 0.0:
            raise ZeroDivisionError('cannot normalize a zero-mass FiniteDist.')
        return FiniteDist(self.__dist / mass, tail_mass=0.0)

    def accurate_pow(self, n):
        '''有更高数值精确性的自卷积n-1次，适用于计算分布概率低于1e-18的情况'''
        ans = self.dist
        for i in range(n-1):
            ans = convolve(ans, self.dist, method='direct')
        return FiniteDist(ans, exp=self.exp * n, var=self.var * n)

    def __add__(self, other: 'FiniteDist') -> 'FiniteDist':
        '''定义的 + 运算符

        返回两个分布值的和，以0位置对齐
        '''
        target_len = max(len(self), len(other))
        return FiniteDist(pad_zero(self.dist, target_len) + pad_zero(other.dist, target_len))

    def __mul__(self, other: Union['FiniteDist', float, int, np.float64, np.int32]) -> 'FiniteDist':
        '''定义的 * 运算符

        如果两个对象都为 FiniteDist 返回两个分布的卷积
        如果其中一个对象为 FiniteDist ，另一个对象为数字，返回分布数乘数字后的 FiniteDist 对象
        '''
        # TODO 研究是否需要对空随机变量进行特判
        if isinstance(other, FiniteDist):
            return FiniteDist(convolve(self.dist, other.dist))
        else:
            return FiniteDist(self.dist * other)
    def __rmul__(self, other: Union['FiniteDist', float, int, np.float64, np.int32]) -> 'FiniteDist':
        return self * other

    def __truediv__(self, other: Union[float, int]) -> 'FiniteDist':
        '''定义的 / 运算符

        返回分布数除数字后的 FiniteDist 对象
        '''
        return FiniteDist(self.dist / other)
    
    def _compute_pow(self, pow_times: int) -> np.ndarray:
        """
        使用快速幂思想计算分布的自卷积
        """
        # 记忆化返回结果
        if pow_times in self._pow_cache:
            # 将使用过的条目移动到末尾，表示最近使用
            self._pow_cache.move_to_end(pow_times)
            return self._pow_cache[pow_times]
        # 特别优化，如果 pow_times-1 已经记录则直接在此基础上卷积，这里不想更复杂的利用缓存的组合实现了
        if pow_times >= 1 and pow_times-1 in self._pow_cache:
            ans = convolve(self.dist, self._pow_cache[pow_times-1])
        # 没有命中缓存则直接用快速幂思想增倍计算
        else:
            ans = np.ones(1)
            if pow_times == 0:
                return ans
            if pow_times == 1:
                return self.dist.copy()
            t = pow_times
            temp = self.dist.copy()
            dist_times = 1
            while t > 0:
                if t % 2 == 1:
                    ans = convolve(ans, temp)
                t = t // 2
                if t > 0:
                    # 利用缓存
                    dist_times *= 2
                    if dist_times in self._pow_cache:
                        # 将使用过的条目移动到末尾，表示最近使用
                        self._pow_cache.move_to_end(dist_times)
                        temp = self._pow_cache[dist_times]
                    else:
                        temp = convolve(temp, temp)
                        self._pow_cache[dist_times] = temp
        
        # 将计算结果加入缓存
        self._pow_cache[pow_times] = ans
        # 如果缓存超过大小限制，移除最久未使用的条目
        while len(self._pow_cache) > self._cache_size:
            self._pow_cache.popitem(last=False)  # 移除最旧的条目
        return ans

    def __pow__(self, pow_times: Union['FiniteDist', int]) -> 'FiniteDist':
        r'''定义的 ** 运算符

        返回分布为 pow_times 个自身分布相卷积的 FiniteDist 对象
        广义乘方扩展到两个 FiniteDist AB的运算，将返回 \sum{A**B[i]} 的值
        '''
        # 整数乘方
        if isinstance(pow_times, int):
            if pow_times < 0:
                raise ValueError("pow_times must be non-negative")
            return FiniteDist(self._compute_pow(pow_times))
        # FiniteDist 乘方
        elif isinstance(pow_times, FiniteDist):
            ans = FiniteDist([0])
            for i in range(len(pow_times)):
                if pow_times[i] == 0:
                    continue
                ans += FiniteDist(pow_times[i] * self._compute_pow(i))
            return ans
        raise TypeError("pow_times must be an integer or FiniteDist")

    def __str__(self) -> str:
        '''字符化为 finite 1D dist 接内容数组'''
        return f"finite 1D dist {self.dist}"

    def __len__(self) -> int:
        return len(self.dist)

if __name__ == "__main__":
    pass
