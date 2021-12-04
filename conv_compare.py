import numpy as np
from scipy import signal
import GGanalysislib as gg
import functools
import time

# 使用numpy的fft进行卷积
def fft_convolve(a, b):
    n = len(a) + len(b) -1
    N = 2 ** (int(np.log2(n))+1)
    A = np.fft.fft(a, N)
    B = np.fft.fft(b, N)
    return np.fft.ifft(A*B)[:n]

## 装饰器
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        begin_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - begin_time
        print('{} 共用时：{} s'.format(func.__name__, run_time))
        return value
    return wrapper

player =  gg.PityGacha()
a = player.get_distribution(1, 90)[1]

num_b = 1000

@timer
def gg_lib_speed(num_b):
    return player.get_distribution(num_b, 90*num_b)[num_b]

@timer
def signal_conv(num_b):
    c = a
    for i in range(num_b-1):
        c = signal.convolve(c, a)
    return c

@timer
def smart_conv(num_b, method='auto'):
    c = np.ones(1)
    temp = a
    t = int(num_b)
    while True:
        if t % 2:
            c = signal.convolve(c, temp, method=method)
        t = int(t/2)
        if t == 0:
            break
        temp = signal.convolve(temp, temp, method=method)
    return c

b = gg_lib_speed(num_b)
c = signal_conv(num_b)
d = smart_conv(num_b)

# d = smart_conv(num_b, method='fft')

print(((b-c)**2).sum())
print(((b-d)**2).sum())