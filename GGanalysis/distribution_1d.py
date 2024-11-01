from typing import Union
import numpy as np
import math
from scipy.signal import convolve
from collections import OrderedDict

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
        dist = np.array(dist)
    return sum(np.arange(len(dist)) * dist)

def calc_variance(dist: Union['FiniteDist', list, np.ndarray]) -> float:
    '''
    计算离散分布列的方差
    '''
    if isinstance(dist, FiniteDist):
        dist = dist.dist
    else:
        dist = np.array(dist)
    use_pulls = np.arange(len(dist))
    exp = sum(use_pulls * dist)
    return sum((use_pulls - exp) ** 2 * dist)

def dist_squeeze(dist: Union['FiniteDist', list, np.ndarray], squeeze_factor) -> 'FiniteDist':
    '''
    按照 squeeze_factor 对分布进行倍数压缩，将压缩部分和存在一起
    '''
    n = math.ceil((len(dist)-1)/squeeze_factor)+1
    new_arr = np.zeros(n, dtype=float)
    new_arr[0] = dist[0]
    for i in range(1, n):
        new_arr[i] = np.sum(dist[(i-1)*squeeze_factor+1:i*squeeze_factor+1])
    return FiniteDist(new_arr)

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
    if len(cdf) == 1:
        # 长度为1 返回必然事件分布
        return FiniteDist([1])
    pdf = np.array(cdf)
    pdf[1:] -= pdf[:-1].copy()
    return FiniteDist(pdf)

def p2dist(pity_p: Union[list, np.ndarray]) -> 'FiniteDist':
    '''
    将保底概率参数转化为分布列
    '''
    # 输入保底参数列表，位置0的概率应为0
    temp = 1
    dist = [0]
    for i in range(1, len(pity_p)):
        dist.append(temp * pity_p[i])
        temp *= (1-pity_p[i])
    return FiniteDist(dist)

def dist2p(dist: Union[np.ndarray, 'FiniteDist']) -> np.ndarray:
    '''
    将分布转换为条件概率表
    '''
    if isinstance(dist, FiniteDist):
        dist = dist.dist
    dist = np.array(dist)
    left_p = np.cumsum(dist[::-1])[::-1]
    return np.divide(dist, left_p, where=left_p!=0, out=np.zeros_like(dist))

def p2exp(pity_p: Union[list, np.ndarray]):
    '''
    对于列表，认为是概率提升表，返回对应分布期望
    '''
    return calc_expectation(p2dist(pity_p))

def p2var(pity_p: Union[list, np.ndarray]):
    '''
    对于列表，认为是概率提升表，返回对应分布方差
    '''
    return calc_variance(p2dist(pity_p))

def pad_zero(dist:np.ndarray, target_len):
    '''
    给 numpy 数组末尾补零至指定长度
    '''
    if target_len <= len(dist):
        return dist
    return np.pad(dist, (0, target_len-len(dist)), 'constant', constant_values=0)

def cut_dist(dist: Union[np.ndarray, 'FiniteDist'], cut_pos):
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

def calc_item_num_dist(dist_list: list['FiniteDist'], pull):
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

class FiniteDist(object):  # 随机事件为有限个数的分布
    '''
    **有限长一维分布**

    用于快速进行有限长一维离散分布的相关计算，创建时通过传入分布数组进行初始化，可以代表一个随机变量。
    本质上是对一个 ``numpy`` 数组的封装，通过利用 FFT、快速幂思想、局部缓存等方法使得各类分布相关的运算能方便高效的进行，有良好的算法复杂度。

    - *** 运算** 定义 ``FiniteDist`` 类型和数值之间的 ``*`` 运算为数值乘，将返回 ``FiniteDist.dist`` 和数值进行数乘后的结果；定义两个 ``FiniteDist`` 类型之间的 ``*`` 运算为卷积，即计算两个随机变量相加的分布。
    - **/ 运算** 定义 ``FiniteDist`` 类型和数值之间的 ``/`` 运算为数值乘，将返回 ``FiniteDist.dist`` 和数值进行数乘后的结果； 定义两个 ``FiniteDist`` 类型之间的 ``+`` 运算为分布叠加，将返回将两个 ``FiniteDist.dist`` 加和后的结果。
    - **\*\* 运算** 定义 ``FiniteDist`` 类型与整数的 ``**`` 运算为自卷积，将返回卷积自身数值次后的结果；``FiniteDist`` 变量A与另一个 ``FiniteDist`` 变量B的 ``**`` 运算为 :math:`\sum_{i=0}^{len(B)}{A^B[i]}`。

    **类属性**

    - ``dist`` : 用列表、numpy数组、FiniteDist表示在自然数位置的分布。若不传入初始化分布默认为全部概率集中在0位置，即离散卷积运算的幺元。
    - ``exp`` ：这个一维分布的期望
    - ``var`` ：这个一维分布的方差
    - ``p_sum`` ：这个一维分布所有位置的概率的和
    - ``entropy_rate`` ：这个一维分布的熵率，即分布的熵除其期望，意义为平均每次尝试的信息量
    - ``randomness_rate`` ：此处定义的随机度为此分布熵率和概率为 :math:`\\frac{1}{exp}`. 的伯努利信源的熵率的比值，越低则说明信息量越低
    
    .. attention:: 
    
        ``FiniteDist`` 类型默认是不可变的对象，不能在外部直接修改 dist 变量，``FiniteDist.dist`` 获得的是一个不可编辑的副本。
        这样设计是为了保证对象内容始终一致，进而能利用缓存信息加速计算。
    
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
    def __init__(self, dist: Union[list, np.ndarray, 'FiniteDist'] = [1]) -> None:
        self._set_dist(dist)

    def _set_dist(self, dist: Union[list, np.ndarray, 'FiniteDist']):
        '''
        用于设置有限长一维分布的值，不推荐使用（类型默认为不可变，修改dist可能会引发问题）
        '''
        # 注意，构造dist时一定要重新创建新的内存空间进行深拷贝
        if dist is None:
            return
        if isinstance(dist, FiniteDist):
            self.__dist = np.array(dist.__dist, dtype=float)
        else:
            if len(np.shape(dist)) > 1:
                raise Exception('Not 1D distribution.')
            self.__dist = np.trim_zeros(np.array(dist, dtype=float), 'b')  # 转化为numpy.ndarray类型并移除分布末尾的0
            if len(self.__dist) == 0:
                self.__dist = np.zeros(1, dtype=float)
        # TODO 仅仅考虑了单线程，没有加入线程锁防止缓存被不同步修改。不过没关系，现在不会用到多线程。
        self._pow_cache = OrderedDict()  # 缓存，记录numpy数组中间值

    @property
    def dist(self):
        # 防止变量被意外修改导致缓存策略出错
        view = self.__dist.view()
        view.flags.writeable = False
        return view

    def __getattr__(self, key):  # 访问未计算的属性时进行计算
        # 基本统计属性
        if key in ['exp', 'var', 'p_sum']:
            self.calc_dist_attribution()
            if key == 'exp':
                return self.exp
            if key == 'var':
                return self.var
            if key == 'p_sum':
                return self.p_sum
        # 累积概率密度函数
        if key == 'cdf':
            self.calc_cdf()
            return self.cdf
        # 熵相关属性
        if key in ['entropy_rate', 'randomness_rate']:
            self.calc_entropy_attribution()
            if key == 'entropy_rate':
                return self.entropy_rate
            if key == 'randomness_rate':
                return self.randomness_rate
    
    def __iter__(self): 
        return iter(self.__dist)
    
    def __setitem__(self, sliced, value: Union[int, float, np.ndarray]) -> None:
        '''将numpy设置切片值的方法应用于 ``dist`` 直接设置分布值'''
        self.__dist[sliced] = value
        self._pow_cache.clear()  # 修改了self._dist需要重置缓存
        
    def __getitem__(self, sliced):
        '''将numpy切片的方法应用于 ``dist`` 直接取得numpy数组切片'''
        return self.__dist[sliced].copy()

    def calc_cdf(self):
        '''将自身分布转为cdf返回'''
        self.cdf = dist2cdf(self.dist)

    def calc_dist_attribution(self, p_error=1e-6) -> None:
        '''
        计算分布的基本属性 ``exp`` ``var`` ``p_sum``
        
        - ``p_error`` ： 容许 ``p_sum`` 和 1 之间的差距，默认为 1e-6
        '''
        self.p_sum = sum(self.dist)
        if abs(self.p_sum-1) > p_error:  # 高于阈值认为概率和不为1
            self.exp = float('nan')
            self.var = float('nan')
            return
        use_pulls = np.arange(self.__len__())
        self.exp = sum(use_pulls * self.dist)
        self.var = sum((use_pulls-self.exp) ** 2 * self.dist)

    def calc_entropy_attribution(self, p_error=1e-6) -> None:
        '''计算分布熵相关属性 ``entropy_rate`` ``randomness_rate``

        - ``p_error`` ： 容许 ``p_sum`` 和 1 之间的差距，默认为 1e-6
        '''
        if abs(self.p_sum-1) > p_error:  # 概率和不为1
            self.entropy_rate = float('nan')
            self.randomness_rate = float('nan')
            return
        # 避免0的对数
        temp = np.zeros(len(self.dist))
        temp[0] = 1
        self.entropy_rate = -sum(self.dist * np.log2(self.dist+temp)) / self.exp
        self.randomness_rate = self.entropy_rate / (-1/self.exp * np.log2(1/self.exp) - (1-1/self.exp) * np.log2(1-1/self.exp))

    def quantile_point(self, quantile_p):
        '''返回分位点位置'''
        return np.searchsorted(self.cdf, quantile_p, side='left')

    def normalized(self) -> 'FiniteDist':
        '''返回分布进行归一化后的结果'''
        return FiniteDist(self.__dist/sum(self.__dist))

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
        '''定义的 ** 运算符
        
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