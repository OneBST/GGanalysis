# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from GGanalysisLite import *
import numpy as np

# 使用finite_dist_1D类进行快速卷积
a = finite_dist_1D([0, 0.25, 0.5, 0.25])    # 从列表初始化分布律
b = finite_dist_1D(np.array([0, 0.5, 0.5]))           # 从numpy数组初始化分布律
c = a * b ** 5                              # c的分布为a的分布卷积5次b的分布
print('a的期望、方差、分布为', a.exp, a.var, a)
print('b的期望、方差、分布为', b.exp, b.var, b)
print('c的期望、方差、分布为', c.exp, c.var, c)

print('分布在类中以numpy数组形式保存', c.dist)

# 计算抽卡分布律 以原神为例
import GGanalysisLite.games.genshin_impact as GI
# 原神角色池的计算
print('角色池在垫了20抽，有大保底的情况下抽3个UP五星抽数的分布')
dist_c = GI.UP_5star_character(item_num=3, pull_state=20, up_guarantee=1)
print('期望为', dist_c.exp, '方差为', dist_c.var)

# 原神武器池的计算
print('武器池池在垫了30抽，有大保底，命定值为1的情况下抽1个UP五星抽数的分布')
dist_w = GI.UP_5star_EP_weapon(item_num=1, pull_state=30, up_guarantee=1, fate_point=1)
print('期望为', dist_w.exp, '方差为', dist_w.var)

# 联合角色池和武器池
print('在前述条件下抽3个UP五星角色，1个特定UP武器所需抽数分布')
dist_c_w = dist_c * dist_w
print('期望为', dist_c_w.exp, '方差为', dist_c_w.var)
# 需要画图则打开注释
# plt.plot(dist_c_w)
# plt.show()

# 对比玩家运气
dist_c = GI.UP_5star_character(item_num=10)
dist_w = GI.UP_5star_EP_weapon(item_num=3)
print('在同样抽了10个UP五星角色，3个特定UP五星武器的玩家中，仅花费1000抽的玩家排名前', str(round(100*sum((dist_c * dist_w)[:1001]), 2))+'%')