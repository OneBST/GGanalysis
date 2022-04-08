from GGanalysisLite.distribution_1d import *
from typing import Union
from scipy.fftpack import fft,ifft
from scipy.special import comb
from scipy.stats import binom
import numpy as np

# 抽卡层的基类，定义了抽卡层的基本行为
class Gacha_layer:
    def __init__(self) -> None:
        # 此处记录的是初始化信息，不会随forward更新
        self.dist = finite_dist_1D([1])
        self.exp = 0
        self.var = 0
    def __call__(self, input: tuple=None, *args: any, **kwds: any) -> tuple:
        # 返回一个元组 (完整分布, 条件分布)
        return self._forward(input, 1), self._forward(input, 0, *args, **kwds)
    def _forward(self, input, full_mode, *args, **kwds) -> finite_dist_1D:
        # 在这项里进行完整分布和条件分布的计算，返回一个元组 (完整分布，条件分布)
        pass

# 保底抽卡层
class Pity_layer(Gacha_layer):
    def __init__(self, pity_p: Union[list, np.ndarray]) -> None:
        super().__init__()
        self.pity_p = pity_p
        self.dist = p2dist(self.pity_p)
        self.exp = self.dist.exp
        self.var = self.dist.var
    def _forward(self, input, full_mode, pull_state=0) -> finite_dist_1D:
        # 输入为空，本层为第一层，返回初始分布
        if input is None:
            return cut_dist(self.dist, pull_state)
        # 处理累加分布情况
        # f_dist 为完整分布 c_dist 为条件分布 根据工作模式不同进行切换
        f_dist: finite_dist_1D = input[0]
        if full_mode:
            c_dist: finite_dist_1D = input[0]
        else:
            c_dist: finite_dist_1D = input[1]
        # 处理条件叠加分布
        overlay_dist = cut_dist(self.dist, pull_state)
        output_dist = finite_dist_1D([0])  # 获得一个0分布
        output_E = 0  # 叠加后的期望
        output_D = 0  # 叠加后的方差
        for i in range(1, len(overlay_dist)):
            c_i = float(overlay_dist[i])  # 防止类型错乱的缓兵之策 如果 c_i 的类型是 numpy 数组，则 numpy 会接管 finite_dist_1D 定义的运算返回错误的类型
            output_dist += c_i * (c_dist * f_dist ** (i-1))  # 分布累加
            output_E += c_i * (c_dist.exp + (i-1) * f_dist.exp)  # 期望累加
            output_D += c_i * (c_dist.var + (i-1) * f_dist.var + (c_dist.exp + (i-1) * f_dist.exp) ** 2)  # 期望平方累加
        output_D -= output_E ** 2  # 计算得到方差
        output_dist.exp = output_E
        output_dist.var = output_D
        return output_dist

# 伯努利抽卡层
class Bernoulli_layer(Gacha_layer):
    def __init__(self, p, e_error = 1e-8, max_dist_len=1e5) -> None:
        super().__init__()
        self.p = p  # 伯努利试验概率
        self.e_error = e_error  # 期望的误差限制
        self.max_dist_len = max_dist_len
        self.exp = 1/p
        self.var = (p**2-2*p+1)/((1-p)*p**2)
    def _forward(self, input, full_mode) -> finite_dist_1D:
        # 作为第一层（不过一般不会当做第一层吧）
        if input is None:
            test_len = int(self.exp * 10)
            while True:
                x = np.arange(test_len+1)
                dist = self.p * (binom.pmf(0, x-1, self.p))
                dist[0] = 0
                # 误差限足够小则停止
                if abs(calc_expectation(dist)-self.exp)/(self.exp) < self.e_error or test_len > self.max_dist_len:
                    if test_len > self.max_dist_len:
                        print('Warning: distribution is too long!')
                    dist = finite_dist_1D(dist)
                    dist.exp = self.exp
                    dist.var = self.var
                    return dist
                test_len *= 2
        # 作为后续输入层
        f_dist: finite_dist_1D = input[0]
        if full_mode:
            c_dist: finite_dist_1D = input[0]
        else:
            c_dist: finite_dist_1D = input[1]
        output_E = self.p*c_dist.exp + (1-self.p) * (f_dist.exp * self.exp + c_dist.exp)  # 叠加后的期望
        output_D = self._calc_combined_2th_moment(f_dist.exp, c_dist.exp, f_dist.var, c_dist.var) - output_E**2  # 叠加后的方差
        test_len = int(output_E+10*output_D**0.5)
        # print(output_E, output_D)
        while True:
            # print('范围', test_len)
            # 通过频域关系进行计算
            F_f = fft(pad_zero(f_dist, test_len))
            F_c = fft(pad_zero(c_dist, test_len))
            output_dist = (self.p * F_c) / (1 - (1-self.p) * F_f)
            output_dist = finite_dist_1D(abs(ifft(output_dist)))
            # 误差限足够小则停止
            if abs(output_dist.exp-output_E)/output_E < self.e_error or test_len > self.max_dist_len:
                if test_len > self.max_dist_len:
                    print('Warning: distribution is too long!')
                output_dist.exp = output_E
                output_dist.var = output_D
                return output_dist
            test_len *= 2

    def _calc_combined_2th_moment(self, Ea, Eb, Da, Db):
        # 计算联合二阶矩 a表示完整序列 b表示条件序列
        p = self.p
        return (p*(p*(Db+Eb**2)+(1-p)*(Da+Ea**2))+2*(1-p)*Ea*(p*Eb+(1-p)*Ea))/(p**2)
    

# 马尔科夫抽卡层
class Markov_layer(Gacha_layer):
    def __init__(self, M: np.ndarray, p_error = 1e-8) -> None:
        super().__init__()
        # 输入矩阵中，零状态为末态
        self.M = M
        self.p_error = p_error
        self.state_len = self.M.shape[0]
        self.dist = self._get_conditional_dist()
        self.exp = self.dist.exp
        self.var = self.dist.var
    
    # 通过转移矩阵计算分布
    def _get_conditional_dist(self, begin_pos=0):
        # 从一个位置开始的转移的分布情况
        dist = [0]
        X = np.zeros(self.state_len)
        X[begin_pos] = 1
        while True:
            X = np.matmul(self.M, X)
            if sum(X) < self.p_error:
                break
            dist.append(X[0])
            X[0] = 0
        return finite_dist_1D(dist)
    def _forward(self, input, full_mode, begin_pos=0) -> finite_dist_1D:
        # 输入为空，本层为第一层，返回初始分布
        if input is None:
            return self._get_conditional_dist(begin_pos)
        # 处理累加分布情况
        # f_dist 为完整分布 c_dist 为条件分布 根据工作模式不同进行切换
        f_dist: finite_dist_1D = input[0]
        if full_mode:
            c_dist: finite_dist_1D = input[0]
        else:
            c_dist: finite_dist_1D = input[1]
        # 处理条件叠加分布
        overlay_dist = self._get_conditional_dist(begin_pos)
        output_dist = finite_dist_1D([0])  # 获得一个0分布
        output_E = 0  # 叠加后的期望
        output_D = 0  # 叠加后的方差
        for i in range(1, len(overlay_dist)):
            c_i = float(overlay_dist[i])  # 防止类型错乱的缓兵之策
            output_dist += c_i * (c_dist * f_dist ** (i-1))  # 分布累加
            output_E += c_i * (c_dist.exp + (i-1) * f_dist.exp)  # 期望累加
            output_D += c_i * (c_dist.var + (i-1) * f_dist.var + (c_dist.exp + (i-1) * f_dist.exp) ** 2)  # 期望平方累加
        output_D -= output_E ** 2  # 计算得到方差
        output_dist.exp = output_E
        output_dist.var = output_D
        return output_dist


# 集齐道具层
class Coupon_Collector_layer(Gacha_layer):
    def __init__(self, item_types, e_error = 1e-12, max_dist_len=1e5) -> None:
        super().__init__()
        self.types = item_types  # 伯努利试验概率
        self.e_error = e_error  # 期望的误差限制
        self.max_dist_len = max_dist_len
        self.exp = 0  # 所需个数的期望
        for i in range(item_types):
            self.exp += item_types/(item_types-i)
        self.var = 0  # 所需个数的方差
        for i in range(item_types):
            self.var += (1-(item_types-i)/item_types)/(((item_types-i)/item_types) ** 2)
    def _forward(self, input, full_mode, get_types=None) -> finite_dist_1D:
        # 如果没有规定需要多少物品，认为需要集齐物品
        if get_types is None:
            get_types = self.types
        # 作为第一层时，初始分布
        if input is None:
            test_len = int(self.exp * 10)
            while True:
                x = np.arange(test_len+1)
                dist = np.zeros(test_len+1)
                for j in range(get_types):
                    dist += (-1) ** j * comb(get_types-1, j) * ((get_types-j-1) / self.types) ** (x-1)
                dist *= comb(self.types-1, get_types-1)
                dist[:get_types] = 0  # 小于这个值概率为0
                # 误差限足够小则停止
                if abs(calc_expectation(dist)-self.exp)/(self.exp) < self.e_error or test_len > self.max_dist_len:
                    if test_len > self.max_dist_len:
                        print('Warning: distribution is too long!')
                    dist = finite_dist_1D(dist)
                    dist.exp = self.exp
                    dist.var = self.var
                    return dist
                test_len *= 2
        # 作为后续输入层(这部分还没写！！！！！！！！！！！！！！！)
        f_dist: finite_dist_1D = input[0]
        if full_mode:
            c_dist: finite_dist_1D = input[0]
        else:
            c_dist: finite_dist_1D = input[1]
        
        # output_E = self.p*c_dist.exp + (1-self.p) * (f_dist.exp * self.exp + c_dist.exp)  # 叠加后的期望
        # output_D = self._calc_combined_2th_moment(f_dist.exp, c_dist.exp, f_dist.var, c_dist.var) - output_E**2  # 叠加后的方差
        # test_len = int(output_E+10*output_D**0.5)
        # # print(output_E, output_D)
        # while True:
        #     # print('范围', test_len)
        #     # 通过频域关系进行计算
        #     F_f = fft(pad_zero(f_dist, test_len))
        #     F_c = fft(pad_zero(c_dist, test_len))
        #     output_dist = (self.p * F_c) / (1 - (1-self.p) * F_f)
        #     output_dist = finite_dist_1D(abs(ifft(output_dist)))
        #     # 误差限足够小则停止
        #     if abs(output_dist.exp-output_E)/output_E < self.e_error or test_len > self.max_dist_len:
        #         if test_len > self.max_dist_len:
        #             print('Warning: distribution is too long!')
        #         output_dist.exp = output_E
        #         output_dist.var = output_D
        #         return output_dist
        #     test_len *= 2

    # def _calc_combined_2th_moment(self, Ea, Eb, Da, Db):
    #   pass



if __name__ == "__main__":
    # 原神武器池参数
    a = np.zeros(78)
    a[1:63] = 0.007
    a[63:77] = np.arange(1, 15) * 0.07 + 0.007
    a[77] = 1
    
