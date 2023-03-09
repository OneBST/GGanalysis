from typing import Union
import numpy as np
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

def dist2cdf(dist: np.array) -> np.array:
    '''简单封装一下numpy的cumsum'''
    return np.cumsum(dist)

def cdf2dist(cdf: np.ndarray) -> np.ndarray:
    '''将cdf转化为分布'''
    if len(cdf) == 1:
        # 长度为1 返回必然事件分布
        return FiniteDist([1])
    pdf = np.array(cdf)
    pdf[1:] -= pdf[:-1].copy()
    return pdf

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

def cut_dist(dist, cut_pos):
    '''
    切除分布头部并重新进行概率归一化
    '''
    # cut_pos为0则没有进行切除
    if cut_pos == 0:
        return dist
    # 进行了切除后进行归一化
    ans = dist[cut_pos:].copy()
    ans[0] = 0
    return FiniteDist(ans/sum(ans))

class FiniteDist(object):  # 随机事件为有限个数的分布
    '''基础类 有限长一维分布律'''
    def __init__(self, dist: Union[list, np.ndarray, 'FiniteDist'] = [1]) -> None:
        # 注意，构造dist时一定要重新创建新的内存空间进行深拷贝
        if isinstance(dist, FiniteDist):
            self.dist = np.array(dist.dist, dtype=float)
            return
        if len(np.shape(dist)) > 1:
            raise Exception('Not 1D distribution.')
        self.dist = np.array(dist, dtype=float)  # 转化为numpy.ndarray类型
        self.dist = np.trim_zeros(self.dist, 'b')  # 去除尾部的零 TODO 想想怎么处理空分布比较好
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
        # 熵相关属性
        if key in ['entropy_rate', 'randomness_rate']:
            self.calc_entropy_attribution()
            if key == 'entropy_rate':
                return self.entropy_rate
            if key == 'randomness_rate':
                return self.randomness_rate
    
    def __iter__(self): 
        return iter(self.dist)
    def __setitem__(self, sliced, value: Union[int, float]) -> None:
        self.dist[sliced] = value
    def __getitem__(self, sliced):
        return self.dist[sliced]

    def calc_dist_attribution(self, p_error=1e-6) -> None:
        self.p_sum = sum(self.dist)
        if abs(self.p_sum-1) > p_error:  # 高于阈值认为概率和不为1
            self.exp = float('nan')
            self.var = float('nan')
            return
        use_pulls = np.arange(self.__len__())
        self.exp = sum(use_pulls * self.dist)
        self.var = sum((use_pulls-self.exp) ** 2 * self.dist)

    def calc_entropy_attribution(self) -> None:
        if abs(self.p_sum-1) > 1e-12:  # 概率和不为1
            self.entropy_rate = float('nan')
            self.randomness_rate = float('nan')
            return
        # 避免0的对数
        temp = np.zeros(len(self.dist))
        temp[0] = 1
        self.entropy_rate = -sum(self.dist * np.log2(self.dist+temp)) / self.exp
        self.randomness_rate = self.entropy_rate / (-1/self.exp * np.log2(1/self.exp) - (1-1/self.exp) * np.log2(1-1/self.exp))

    def p_normalization(self) -> None:
        # 分布概率归一
        self.dist = self.dist/sum(self.dist)
        self.calc_dist_attribution()

    def __add__(self, other: 'FiniteDist') -> 'FiniteDist':
        target_len = max(len(self), len(other))
        return FiniteDist(pad_zero(self.dist, target_len) + pad_zero(other.dist, target_len))

    def __mul__(self, other: Union['FiniteDist', float, int, np.float64, np.int32]) -> 'FiniteDist':
        # TODO 研究是否需要对空随机变量进行特判
        if isinstance(other, FiniteDist):
            return FiniteDist(convolve(self.dist, other.dist))
        else:
            return FiniteDist(self.dist * other)
    def __rmul__(self, other: Union['FiniteDist', float, int, np.float64, np.int32]) -> 'FiniteDist':
        return self * other

    def __truediv__(self, other: Union[float, int]) -> 'FiniteDist':
        return FiniteDist(self.dist / other)
    def __pow__(self, times_: int) -> 'FiniteDist':
        ans = np.ones(1)
        if times_ == 0:
            return FiniteDist(ans)
        if times_ == 1:
            return self
        t = times_
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
        return f"finite 1D dist {self.dist}"

    def __len__(self) -> int:
        return len(self.dist)

if __name__ == "__main__":
    pass