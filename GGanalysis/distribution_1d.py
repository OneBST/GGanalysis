from typing import Union
import numpy as np
import math
from scipy.signal import convolve

def linear_p_increase(base_p=0.01, pity_begin=100, step=1, hard_pity=100):
    '''
    建立线性递增模型的保底参数
    '''
    ans = np.zeros(hard_pity+1)
    ans[1:pity_begin] = base_p
    ans[pity_begin:hard_pity+1] = np.arange(1, hard_pity-pity_begin+2) * step + base_p
    return np.minimum(ans, 1)

def calc_expectation(dist: Union['FiniteDist', list, np.ndarray]) -> float:
    '''
    输入离散分布列计算期望
    '''
    if isinstance(dist, FiniteDist):
        dist = dist.dist
    else:
        dist = np.array(dist)
    return sum(np.arange(len(dist)) * dist)

def calc_variance(dist: Union['FiniteDist', list, np.ndarray]) -> float:
    '''
    输入离散分布列计算方差
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
    '''简单封装一下numpy的cumsum'''
    if isinstance(dist, FiniteDist):
        return np.cumsum(dist.dist)
    return np.cumsum(dist)

def cdf2dist(cdf: np.ndarray) -> 'FiniteDist':
    '''将cdf转化为分布'''
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
    '''将分布转换为条件概率表'''
    if isinstance(dist, FiniteDist):
        dist = dist.dist
    dist = np.array(dist)
    left_p = np.cumsum(dist[::-1])[::-1]
    return np.divide(dist, left_p, where=left_p!=0, out=np.zeros_like(dist))

def p2exp(pity_p: Union[list, np.ndarray]):
    '''
    对于列表，认为是概率提升表，返回对应期望
    '''
    return calc_expectation(p2dist(pity_p))

def p2var(pity_p: Union[list, np.ndarray]):
    '''
    对于列表，认为是概率提升表，返回对应方差
    '''
    return calc_variance(p2dist(pity_p))

def table2matrix(state_num, state_trans):
    '''
    将邻接表变为邻接矩阵

    构造 state_num 和 state_trans 示例
    Epitomized Path & Fate Points
    state_num = {'get':0, 'fate1UP':1, 'fate1':2, 'fate2':3}
    state_trans = [
        ['get', 'get', 0.375],
        ['get', 'fate1UP', 0.375],
        ['get', 'fate1', 0.25],
        ['fate1UP', 'get', 0.375],
        ['fate1UP', 'fate2', 1-0.375],
        ['fate1', 'get', 0.5],
        ['fate1', 'fate2', 0.5],
        ['fate2', 'get', 1]
    ]
    '''
    M = np.zeros((len(state_num), len(state_num)))
    for name_a, name_b, p in state_trans:
        a = state_num[name_a]
        b = state_num[name_b]
        M[b][a] = p
    # 检查每个节点出口概率和是否为1, 但这个并不是特别广义
    '''
    for index, element in enumerate(np.sum(M, axis=0)):
        if element != 1:
            raise Warning('The sum of probabilities is not 1 at position '+str(index))
    '''
    return M

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
    '''**有限长一维分布**

    - ``dist`` : 用列表、numpy数组、FiniteDist表示在自然数位置的分布。默认为全部概率集中在0位置，即离散卷积运算的幺元。
    - ``exp`` ：这个一维分布的期望
    - ``var`` ：这个一维分布的方差
    - ``p_sum`` ：这个一维分布所有位置的概率的和
    - ``entropy_rate`` ：这个一维分布的熵率，即分布的熵除其期望，意义为平均每次尝试的信息量
    - ``randomness_rate`` ：此处定义的随机度为此分布熵率和概率为 :math:`\\frac{1}{exp}`. 的伯努利信源的熵率的比值，越低则说明信息量越低
    
    '''
    def __init__(self, dist: Union[list, np.ndarray, 'FiniteDist'] = [1]) -> None:
        # 注意，构造dist时一定要重新创建新的内存空间进行深拷贝
        if isinstance(dist, FiniteDist):
            self.dist = np.array(dist.dist, dtype=float)
            return
        if len(np.shape(dist)) > 1:
            raise Exception('Not 1D distribution.')
        self.dist = np.array(dist, dtype=float)  # 转化为numpy.ndarray类型
        if len(self.dist) == 0:
            self.dist = np.zeros(1, dtype=float)

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
        return iter(self.dist)
    def __setitem__(self, sliced, value: Union[int, float, np.ndarray]) -> None:
        '''将numpy设置切片值的方法应用于 ``dist`` 直接设置分布值'''
        self.dist[sliced] = value
    def __getitem__(self, sliced):
        '''将numpy切片的方法应用于 ``dist`` 直接取得numpy数组切片'''
        return self.dist[sliced]

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

    def p_normalization(self) -> None:
        # 分布概率归一
        self.dist = self.dist/sum(self.dist)
        self.calc_dist_attribution()

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
    def __pow__(self, pow_times: int) -> 'FiniteDist':
        '''定义的 ** 运算符
        
        返回分布为 pow_times 个自身分布相卷积的 FiniteDist 对象
        '''
        ans = np.ones(1)
        if pow_times == 0:
            return FiniteDist(ans)
        if pow_times == 1:
            return self
        t = pow_times
        temp = self.dist
        while True:
            if t % 2:
                ans = convolve(ans, temp)
            t = t//2
            if t == 0:
                break
            temp = convolve(temp, temp)
        return FiniteDist(ans)

    def __str__(self) -> str:
        '''字符化为 finite 1D dist 接内容数组'''
        return f"finite 1D dist {self.dist}"

    def __len__(self) -> int:
        return len(self.dist)

if __name__ == "__main__":
    pass