import GGanalysislib as gg
import GGanalysisLite as ggl
import time

test_num = 10  # 测试个数
# 对比dll里写的O(N^2)的DP和python里写的O(N^2logN)算法谁更快

# python版本
begin = time.time()
lite = ggl.Pity5starCommon()
ans1 = lite.calc_distribution_1_to_n(test_num)[test_num]
end = time.time()
print('version:Lite time = ', end-begin)

# dll版本
begin = time.time()
dll = gg.PityGacha()
ans2 = dll.get_distribution(test_num, test_num*90)[test_num]
end = time.time()
print('version:dll time = ', end-begin)

# 对比差异
print(sum(abs(ans1-ans2)))