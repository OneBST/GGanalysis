from GGanalysis.distribution_1d import *
from typing import Union
from scipy.fftpack import fft,ifft
from scipy.special import comb
from scipy.stats import binom
from functools import lru_cache
import numpy as np
import warnings

class GachaLayer(object):
    # 抽卡层的基类，定义了抽卡层的基本行为
    def __init__(self) -> None:
        # 此处记录的是初始化信息，不会随forward更新
        self.dist = FiniteDist([1])
        self.exp = 0
        self.var = 0
    def __call__(self, input: tuple=None, *args: any, **kwds: any) -> tuple[FiniteDist, FiniteDist]:
        # 返回一个元组 (完整分布, 条件分布)
        return self._forward(input, 1), self._forward(input, 0, *args, **kwds)
    @lru_cache
    def _forward(self, input, full_mode, *args, **kwds) -> FiniteDist:
        # 根据full_mode在这项里进行完整分布或条件分布的计算，返回一个分布
        pass
    def __str__(self) -> str:
        return "Gacha Layer\n"

class PityLayer(GachaLayer):
    # 保底抽卡层
    def __init__(self, pity_info: Union[list, np.ndarray, FiniteDist]) -> None:
        super().__init__()
        if isinstance(pity_info, FiniteDist):
            # 输入为分布列
            self.dist = pity_info
        else:
            # 输入为保底概率表
            self.pity_p = pity_info
            self.dist = p2dist(self.pity_p)
        self.exp = self.dist.exp
        self.var = self.dist.var
    def __str__(self) -> str:
        return f"Pity Layer E={round(self.exp, 2)} V={round(self.var, 2)}"
    
    @lru_cache
    def _forward(self, input, full_mode, item_pity=0) -> FiniteDist:
        # 输入为空，本层为第一层，返回初始分布
        if input is None:
            return FiniteDist(cut_dist(self.dist, item_pity))
        # 处理累加分布情况
        # f_dist 为完整分布 c_dist 为条件分布 根据工作模式不同进行切换
        f_dist: FiniteDist = input[0]
        if full_mode:
            c_dist: FiniteDist = input[0]
        else:
            c_dist: FiniteDist = input[1]
        # 处理条件叠加分布
        overlay_dist = FiniteDist(cut_dist(self.dist, item_pity))
        output_dist = FiniteDist([0])  # 获得一个空分布
        output_E = 0  # 叠加后的期望
        output_D = 0  # 叠加后的方差
        temp_dist = FiniteDist([1]) # 用于优化计算量
        # 对0位置特殊处理
        output_dist += float(overlay_dist[0]) * temp_dist
        for i in range(1, len(overlay_dist)):
            c_i = float(overlay_dist[i])  # 防止类型错乱的缓兵之策 如果 c_i 的类型是 numpy 数组，则 numpy 会接管 finite_dist_1D 定义的运算返回错误的类型
            output_dist += c_i * (c_dist * temp_dist)  # 分布累加
            temp_dist = temp_dist * f_dist
            output_E += c_i * (c_dist.exp + (i-1) * f_dist.exp)  # 期望累加
            output_D += c_i * (c_dist.var + (i-1) * f_dist.var + (c_dist.exp + (i-1) * f_dist.exp) ** 2)  # 期望平方累加
        output_D -= output_E ** 2  # 计算得到方差
        output_dist.exp = output_E
        output_dist.var = output_D
        return output_dist

class BernoulliLayer(GachaLayer):
    # 伯努利抽卡层
    def __init__(self, p, e_error = 1e-8, max_dist_len=1e5) -> None:
        super().__init__()
        self.p = p  # 伯努利试验概率
        self.e_error = e_error  # 期望的误差限制
        self.max_dist_len = max_dist_len
        self.exp = 1/p
        if p == 1:
            self.var = 0
        else:
            self.var = (p**2-2*p+1)/((1-p)*p**2)
    def __str__(self) -> str:
        return f"Bernoulli Layer E={round(self.exp, 2)} V={round(self.var, 2)}"
    
    @lru_cache
    def _forward(self, input, full_mode) -> FiniteDist:
        # 作为第一层（不过一般不会当做第一层吧）
        if input is None:
            if self.p == 1:
                return FiniteDist([0, 1])
            test_len = int(self.exp * 10)
            while True:
                x = np.arange(test_len+1)
                dist = self.p * (binom.pmf(0, x-1, self.p))
                dist[0] = 0
                # 误差限足够小则停止
                calc_error = abs(calc_expectation(dist)-self.exp)/(self.exp)
                if calc_error < self.e_error or test_len > self.max_dist_len:
                    if test_len > self.max_dist_len:
                        print('Warning: distribution is too long! len:', test_len, 'Error:', calc_error)
                    dist = FiniteDist(dist)
                    dist.exp = self.exp
                    dist.var = self.var
                    return dist
                test_len *= 2
        # 作为后续输入层
        f_dist: FiniteDist = input[0]
        if full_mode:
            c_dist: FiniteDist = input[0]
        else:
            c_dist: FiniteDist = input[1]
        # 概率为 1 等价于什么也没干
        if self.p == 1:
            return c_dist
        output_E = self.p*c_dist.exp + (1-self.p) * (f_dist.exp * self.exp + c_dist.exp)  # 叠加后的期望
        output_D = self._calc_combined_2nd_moment(f_dist.exp, c_dist.exp, f_dist.var, c_dist.var) - output_E**2  # 叠加后的方差
        test_len = int(output_E+10*output_D**0.5)
        while True:
            # 通过频域关系进行计算
            F_f = fft(pad_zero(f_dist.dist, test_len))
            F_c = fft(pad_zero(c_dist.dist, test_len))
            output_dist = (self.p * F_c) / (1 - (1-self.p) * F_f)
            output_dist = FiniteDist(abs(ifft(output_dist)))
            # 解决输出位置0处不是0的问题
            output_dist[0] = 0
            # 误差限足够小则停止
            calc_error = abs(calc_expectation(output_dist)-output_E)/output_E
            if calc_error < self.e_error or test_len > self.max_dist_len:
                if test_len > self.max_dist_len:
                    print('Warning: distribution is too long! len:', test_len, 'Error:', calc_error)
                output_dist.exp = output_E
                output_dist.var = output_D
                return output_dist
            test_len *= 2

    def _calc_combined_2nd_moment(self, Ea, Eb, Da, Db):
        # 计算联合二阶矩 a表示完整序列 b表示条件序列
        p = self.p
        return (p*(p*(Db+Eb**2)+(1-p)*(Da+Ea**2))+2*(1-p)*Ea*(p*Eb+(1-p)*Ea))/(p**2)
    
class MarkovLayer(GachaLayer):
    # 马尔科夫抽卡层,对于不能在有限次数内移动到目标态的情况采用截断的方法处理
    def __init__(self, M: np.ndarray, p_error=1e-8) -> None:
        super().__init__()
        # 输入矩阵中，零状态为末态
        self.M = M
        self.p_error = p_error
        self.state_len = self.M.shape[0]
        self.dist = self._get_conditional_dist()
        self.exp = self.dist.exp
        self.var = self.dist.var
    def __str__(self) -> str:
        return f"Markov Layer E={round(self.exp, 2)} V={round(self.var, 2)}"
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
        return FiniteDist(dist)

    @lru_cache
    def _forward(self, input, full_mode, begin_pos=0) -> FiniteDist:
        # 输入为空，本层为第一层，返回初始分布
        if input is None:
            return self._get_conditional_dist(begin_pos)
        # 处理累加分布情况
        # f_dist 为完整分布 c_dist 为条件分布 根据工作模式不同进行切换
        f_dist: FiniteDist = input[0]
        if full_mode:
            c_dist: FiniteDist = input[0]
        else:
            c_dist: FiniteDist = input[1]
        # 处理条件叠加分布
        overlay_dist = self._get_conditional_dist(begin_pos)
        output_dist = FiniteDist([0])  # 获得一个0分布
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

class DynamicProgrammingLayer(GachaLayer):
    '''DP抽卡层 传入DP函数以定义，注意DP函数需要定义默认值'''
    # TODO 完善这个抽卡层并进行测试，目前还是没有经过测试的不可用阶段，其功能和接口也尚未评估
    def __init__(self, dp_function) -> None:
        super().__init__()
        self.dp_function = dp_function
    def __str__(self) -> str:
        return f"DP Layer {self.dp_function}"
    
    @lru_cache
    def _forward(self, input, full_mode, *args: any, **kwds: any) -> FiniteDist:
        # 以下代码修改自 PityLayer
        # 输入为空，本层为第一层，返回初始分布
        if input is None:
            return FiniteDist(self.dp_function(*args, **kwds))
        # 处理累加分布情况
        # f_dist 为完整分布 c_dist 为条件分布 根据工作模式不同进行切换
        f_dist: FiniteDist = input[0]
        if full_mode:
            c_dist: FiniteDist = input[0]
        else:
            c_dist: FiniteDist = input[1]
        # 处理条件叠加分布
        overlay_dist = FiniteDist(self.dp_function(*args, **kwds))
        output_dist = FiniteDist([0])  # 获得一个0分布
        output_E = 0  # 叠加后的期望
        output_D = 0  # 叠加后的方差
        temp_dist = FiniteDist([1]) # 用于优化计算量
        for i in range(1, len(overlay_dist)):
            c_i = float(overlay_dist[i])  # 防止类型错乱的缓兵之策 如果 c_i 的类型是 numpy 数组，则 numpy 会接管 finite_dist_1D 定义的运算返回错误的类型
            output_dist += c_i * (c_dist * temp_dist)  # 分布累加
            temp_dist = temp_dist * f_dist
            output_E += c_i * (c_dist.exp + (i-1) * f_dist.exp)  # 期望累加
            output_D += c_i * (c_dist.var + (i-1) * f_dist.var + (c_dist.exp + (i-1) * f_dist.exp) ** 2)  # 期望平方累加
        output_D -= output_E ** 2  # 计算得到方差
        output_dist.exp = output_E
        output_dist.var = output_D
        return output_dist

class CouponCollectorLayer(GachaLayer):
    '''均等概率集齐道具层'''
    # 集齐道具层写完了，但是还没有测试
    # 集齐道具层，一般用于最后一层 如果想要实现集齐k种（不足总种类）后继续进入下一层的模型，需要在初始化时给出 target_types
    def __init__(self, item_types, target_types=None, e_error=1e-6, max_dist_len=1e5) -> None:
        super().__init__()
        self.types = item_types # 道具种类
        self.target_types = target_types  # 目标抽数，用于多次使用时
        self.e_error = e_error  # 期望的误差限制
        self.max_dist_len = max_dist_len
        if target_types is None:
            self.exp, self.var = self._calc_exp_var(0, self.types)
        else:
            self.exp, self.var = self._calc_exp_var(0, self.target_types)
    def __str__(self) -> str:
        return f"Coupon Collector Layer E={round(self.exp, 2)} V={round(self.var, 2)}"
    # 计算抽k类物品的期望和方差
    def _calc_exp_var(self, initial_types, target_types):
        ans_exp = 0
        ans_var = 0
        # 默认为抽齐
        for i in range(initial_types+1, target_types+1):
            ans_exp += self.types / (self.types-i+1)
            ans_var += (i-1) * self.types / (self.types-i+1) ** 2
        return ans_exp, ans_var
    
    # 下面几个函数是用来计算系数的，方便代码阅读
    # 计算统一乘的系数
    def _calc_coefficient_1(self, initial_types, target_types):
        return comb(self.types-initial_types, target_types-initial_types-1) * (self.types-target_types+1) / self.types
    # 计算内部乘的系数
    def _calc_coefficient_2(self, i, j, initial_types, target_types):
        return (-1)**j * comb(initial_types, i) * comb(target_types-initial_types+i-1, j)
    # 计算乘方内部系数
    def _calc_coefficient_3(self, i, j, initial_types, target_types):
        return (target_types-initial_types+i-j-1) / self.types

    @lru_cache
    def _forward(self, input, full_mode, initial_types=0, target_types=None) -> FiniteDist:
        # 便于和推导统一的标记，同时自动识别应该如何处理
        if target_types is not None:
            k = target_types
        elif self.target_types is not None:
            k = self.target_types
        else:
            k = self.types
        # 如果目标种类为1时，计算会出现问题（方差出现负数），需要特殊处理
        # 这里的处理是否恰当，还需要研判，没有检查方差计算部分，可能是n=1是公式没考虑的特殊情况
        if k == 1:
            warnings.warn(
                'Note target_types equals to 1, there no need to add this layer!', 
                DeprecationWarning)
            if full_mode:
                return input[0]
            else:
                return input[1]
        # 初始就有的种类
        a = initial_types
        # 作为第一层时，给出初始分布
        if input is None:
            output_E, output_D = self._calc_exp_var(a, k)
            test_len = int(output_E + 10 * output_D ** 0.5)
            while True:
                x = np.arange(test_len+1)
                dist = np.zeros(test_len+1)
                C1 = self._calc_coefficient_1(a, k)
                for i in range(a+1):
                    for j in range(k-a+i):
                        C2 = self._calc_coefficient_2(i, j, a, k)
                        C3 = self._calc_coefficient_3(i, j, a, k)
                        dist[k-a:] += C2 * C3 ** (x[k-a:]-1)
                dist *= C1
                # 误差限足够小则停止
                calc_error = abs(calc_expectation(dist)-output_E)/output_E
                if calc_error < self.e_error or test_len > self.max_dist_len:
                    if test_len > self.max_dist_len:
                        print('Warning: distribution is too long! len:', test_len, 'Error:', calc_error)
                    dist = FiniteDist(dist)
                    dist.exp = output_E
                    dist.var = output_D
                    return dist
                test_len *= 2

        # 作为后续输入层
        # f_dist 为完整分布 c_dist 为条件分布 根据工作模式不同进行切换
        f_dist: FiniteDist = input[0]
        if full_mode:
            c_dist: FiniteDist = input[0]
        else:
            c_dist: FiniteDist = input[1]
        
        output_E = c_dist.exp + (self._calc_exp_var(a, k)[0]-1) * f_dist.exp  # 叠加后的期望
        # 计算叠加后的方差
        def calc_combined_output_2nd_moment():
            # 中间所用的一阶与二阶矩
            Ef = f_dist.exp
            E2f = f_dist.exp**2 + f_dist.var
            Ec = c_dist.exp
            E2c = c_dist.exp**2 + c_dist.var
            # C1系数
            C1 = self._calc_coefficient_1(a, k)
            # 临时标记指数
            n = k-a
            ans_2moment = 0
            ans_temp = 0
            # 累加计算
            for i in range(a+1):
                for j in range(k-a+i):
                    # 防止遇到除0情况
                    if k-a+i-j-1 == 0:
                        continue
                    C2 = self._calc_coefficient_2(i, j, a, k)
                    C3 = self._calc_coefficient_3(i, j, a, k)
                    ans_ij = ((C3**n) / (C3-1)**3) * \
                            ((C3-1)*(-2*Ef*Ec*((n-2)*C3-n+1)+(1-C3)*E2c)+\
                            ((1-C3)*E2f*((n-2)*C3-n+1)-(Ef**2)*((n**2-5*n+6)*(C3**2)-2*(n**2-4*n+3)*C3+n**2-3*n+2)))
                    
                    ans_temp += C1*C2*(C3**n)*(Ef*((n-2)*C3-n+1)+(C3-1)*Ec)/(C3-1)**2
                    ans_2moment += C2 / C3 * ans_ij
            ans_2moment *= C1
            return ans_2moment
        output_D = calc_combined_output_2nd_moment() - output_E**2  # 叠加后的方差
        test_len = int(output_E+10*output_D**0.5)
        while True:
            # 通过频域关系进行计算
            F_f = fft(pad_zero(f_dist.dist, test_len))
            F_c = fft(pad_zero(c_dist.dist, test_len))
            output_dist = 0
            C1 = self._calc_coefficient_1(a, k)
            # 预计算F_f ^ (k-a)减少计算量
            buff_F_f_multi = F_f ** (k-a)
            for i in range(a+1):
                for j in range(k-a+i):
                    # 防止遇到除0情况
                    if k-a+i-j-1 == 0:
                        continue
                    C2 = self._calc_coefficient_2(i, j, a, k)
                    C3 = self._calc_coefficient_3(i, j, a, k)
                    output_dist += (C2/C3) *  (C3**(k-a)*buff_F_f_multi / (1-C3*F_f))
            output_dist = output_dist * C1 * F_c / F_f
            output_dist = FiniteDist(abs(ifft(output_dist)))
            # 解决输出位置0处不是0的问题
            output_dist[0] = 0
            # 误差限足够小则停止
            calc_error = abs(output_dist.exp-output_E)/output_E
            if calc_error < self.e_error or test_len > self.max_dist_len:
                if test_len > self.max_dist_len:
                    print('Warning: distribution is too long! len:', test_len, 'Error:', calc_error)
                output_dist.exp = output_E
                output_dist.var = output_D
                return output_dist
            test_len *= 2

if __name__ == "__main__":
    pass
