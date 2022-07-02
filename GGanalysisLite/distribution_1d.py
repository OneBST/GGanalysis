from typing import Union
import numpy as np
from scipy.signal import convolve

# 建立线性递增模型的保底参数
def linear_p_increase(base_p=0.01, pity_begin=100, step=1, hard_pity=100):
    ans = np.zeros(hard_pity+1)
    ans[1:pity_begin] = base_p
    ans[pity_begin:hard_pity+1] = np.arange(1, hard_pity-pity_begin+2) * step + base_p
    return np.minimum(ans, 1)

# 输入离散分布列计算期望
def calc_expectation(dist: Union['finite_dist_1D', list, np.ndarray]) -> float:
    if isinstance(dist, finite_dist_1D):
        dist = dist.dist
    else:
        dist = np.array(dist)
    return sum(np.arange(len(dist)) * dist)

# 输入离散分布列计算方差
def calc_variance(dist: Union['finite_dist_1D', list, np.ndarray]) -> float:
    if isinstance(dist, finite_dist_1D):
        dist = dist.dist
    else:
        dist = np.array(dist)
    use_pulls = np.arange(len(dist))
    exp = sum(use_pulls * dist)
    return sum((use_pulls - exp) ** 2 * dist)

# 将保底概率参数转化为分布列
def p2dist(pity_p: Union[list, np.ndarray]) -> 'finite_dist_1D':
    # 输入保底参数列表，位置0的概率应为0
    temp = 1
    dist = [0]
    for i in range(1, len(pity_p)):
        dist.append(temp * pity_p[i])
        temp *= (1-pity_p[i])
    return finite_dist_1D(dist)

# 将邻接表变为邻接矩阵
def table2matrix(state_num, state_trans):
    '''
    # 构造 state_num 和 state_trans 示例
    # Epitomized Path & Fate Points
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
    return M

# 给 numpy 数组末尾补零至指定长度
def pad_zero(dist, target_len):
    if target_len <= len(dist):
        return dist
    return np.pad(dist, (0, target_len-len(dist)), 'constant', constant_values=0)

# 切除分布尾部并重新进行概率归一化
def cut_dist(dist, cut_pos):
    # cut_pos为0则没有进行切除
    if cut_pos == 0:
        return dist
    # 进行了切除后进行归一化
    ans = dist[cut_pos:].copy()
    ans[0] = 0
    return finite_dist_1D(ans/sum(ans))

# 基础类 有限长一维分布律
class finite_dist_1D:  # 随机事件为有限个数的分布
    def __init__(self, dist: Union[list, np.ndarray] = [1]) -> None:
        if len(np.shape(dist)) > 1:
            raise Exception('Not 1D distribution.')
        self.dist = np.array(dist)  # 转化为numpy.ndarray类型
        
    def __getattr__(self, key):  # 访问未计算的属性时进行计算
        # if 'array' in key:  # 不返回numpy数组，强制使用类中定义运算，这里可能需要修改 和numpy相关的就蛋疼
        if key in ['exp', 'var', 'p_sum']:
            self.calc_dist_attribution()
            if key == 'exp':
                return self.exp
            if key == 'var':
                return self.var
            if key == 'p_sum':
                return self.p_sum
        # 还是决定注释掉，不自动进行numpy变换，需要手工.dist
        # return super().__getattr__(key)
    
    def __iter__(self): 
        return iter(self.dist)
    def __setitem__(self, sliced, value: Union[int, float]) -> None:
        self.dist[sliced] = value
    def __getitem__(self, sliced):
        return self.dist[sliced]

    def calc_dist_attribution(self) -> None:
        self.p_sum = sum(self.dist)
        if abs(self.p_sum-1) > 1e-12:  # 不是标准分布
            self.exp = float('nan')
            self.var = float('nan')
            return
        use_pulls = np.arange(self.__len__())
        self.exp = sum(use_pulls * self.dist)
        self.var = sum((use_pulls-self.exp) ** 2 * self.dist)

    def p_normalization(self):  # 分布概率归一
        self.dist = self.dist/sum(self.dist)
        self.calc_dist_attribution()

    def __add__(self, other: 'finite_dist_1D') -> 'finite_dist_1D':
        target_len = max(len(self), len(other))
        return finite_dist_1D(pad_zero(self.dist, target_len) + pad_zero(other.dist, target_len))
    def __radd__(self, other: 'finite_dist_1D') -> 'finite_dist_1D':
        return self + other

    def __mul__(self, other: Union['finite_dist_1D', float, int, np.float64, np.int32]) -> 'finite_dist_1D':
        if isinstance(other, finite_dist_1D):
            return finite_dist_1D(convolve(self.dist, other.dist))
        else:
            return finite_dist_1D(self.dist * other)
    def __rmul__(self, other: Union['finite_dist_1D', float, int, np.float64, np.int32]) -> 'finite_dist_1D':
        return self * other

    def __truediv__(self, other: Union[float, int]) -> 'finite_dist_1D':
        ans = finite_dist_1D()
        ans.dist = self.dist / other
        return ans
    def __pow__(self, times_: int) -> 'finite_dist_1D':
        ans = np.ones(1)
        if times_ == 0:
            return finite_dist_1D(ans)
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
        return finite_dist_1D(ans)

    def __str__(self) -> str:
        return f"finite 1D dist {self.dist}"

    def __len__(self) -> int:
        return len(self.dist)

if __name__ == "__main__":
    pass