import GGanalysis.games.arknights_endfield as AKE
import numpy as np
from matplotlib import pyplot as plt

# 计算在垫了i抽情况下，轮换到下个卡池清除120保底时获取下一个UP6星的条件期望
condition_exp = np.zeros(80)
raw_exp = AKE.up_6star_first_character(1).exp
for i in range(80):
    dist = AKE.up_6star_first_character(1, item_pity=i, single_up_pity=0)
    condition_exp[i] = dist.exp

# 从无保底开始抽n抽的情况下，当前垫了几抽分布的递推，由于没有到概率提升段，所以很简单
M = np.zeros((61, 61))
M[0, 0] = 1
for i in range(1, 61):
    # 获得6星情况
    M[i, 0] = sum(M[i-1, :]) * AKE.PITY_6STAR[1]
    for j in range(1, i+1):
        # 当前已经垫了多少抽
        M[i, j] = M[i-1, j-1] * (1-AKE.PITY_6STAR[1])

# 计算策略导致的期望垫抽损失
exp_loss = np.zeros(61)
for i in range(1, 61):
    exp_loss[i] = i + sum(M[i, :] * condition_exp[:61]) - raw_exp
# 这里的结论是，垫15抽时损失恰好低于10抽。也即是如果要角色全收集，每次第45-59抽出UP时垫到60都是抽数期望上不亏的。

plt.plot(exp_loss)
plt.hlines(10, xmin=0, xmax=len(exp_loss)-1)
plt.show()

'''
垫抽损失（Padding Loss）的严格定义
相对于从零开始抽的基准期望，计划垫入i抽，再去下一个卡池继续抽目标，多出来的期望抽数
L是垫抽损失 i 是已经垫入的抽数（确定量） u_i 是计划垫i抽后，进入下池后还需要的随机抽数期望 u 是原本所需的基准期望
L = u_i + i - u

条件垫抽损失
相对于从零开始抽的基准期望，当前已经垫了i抽，再去下一个卡池继续抽目标，多出来的期望抽数
'''