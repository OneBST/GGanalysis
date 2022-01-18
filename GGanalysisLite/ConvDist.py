import numpy as np
from scipy import signal

# 快速幂+FFT卷积
def smart_conv(dist , item_num, method='auto'):
    ans = np.ones(1)
    temp = dist
    t = item_num
    if t == 0:
        return ans
    if t == 1:
        return dist
    while True:
        if t % 2:
            ans = signal.convolve(ans, temp, method=method)
        t = t//2
        if t == 0:
            break
        temp = signal.convolve(temp, temp, method=method)
    return ans

# 快速计算1-n次的卷积结果，以列表形式返回
def smart_conv_1_to_n(dist, item_num, method='auto'):
    def lowbit(x):
        return x&-x
    # 零处占位
    ans = [[1], dist]
    for i in range(2, item_num+1):
        if lowbit(i) == i:
            ans.append(signal.convolve(ans[i//2], ans[i//2], method=method))
        else:
            ans.append(signal.convolve(ans[lowbit(i)], ans[i-lowbit(i)], method=method))
    return ans

# 根据分布计算期望
def calc_expectation(dist):
    temp = np.arange(0, len(dist), 1, dtype=float)
    return (dist*temp).sum()

if __name__ == '__main__':
    pass