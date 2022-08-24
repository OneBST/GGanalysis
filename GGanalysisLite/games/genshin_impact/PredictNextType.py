import numpy as np
import GGanalysisLite as ggl
import GGanalysisLite.games.genshin_impact as GI
from matplotlib import pyplot as plt

# 计算常驻类别概率
def calc_type_P(pull_state, not_met):
    # c = ggl.Pity5starCommon()
    # dist = c.conditional_distribution(1, pull_state)
    dist = GI.Common_5star(1, pull_state=pull_state)
    # 计算单抽类别概率
    def pull_type_P(pull_num):
        A = 30
        B = 30
        if pull_num > 147:
            B += 300*(pull_num-147)
        # 轮盘选择法截断了
        if A+B > 10000:
            # print(B, pull_num)
            return min(10000, B)/10000
        # 轮盘选择法没有截断
        return B/(A+B)
    ans_P = 0
    for i in range(1, len(dist)):
        ans_P += pull_type_P(not_met+i)*dist[i]
    return ans_P


# 绘制概率图
x = np.linspace(0, 180, 181)
y = np.linspace(0, 89, 90)
X, Y = np.meshgrid(x, y)
p_map = np.zeros((90, 181), dtype=float)
for i in range(0, 90):
    for j in range(0, 181):
        p_map[i][j] = calc_type_P(i, j)

X = X[:, 50:]
Y = Y[:, 50:]
p_map = p_map[:, 50:]

# 绘图部分
fig, ax = plt.subplots(dpi=150, figsize=(15, 5))
ax.set_xticks(range(50, 190, 5))
ax.grid(visible=True, which='major', color='k', alpha=0.2, linewidth=1)
ax.grid(visible=True, which='minor', color='k', alpha=0.2, linewidth=0.5)
plt.minorticks_on()
ax.contourf(X, Y, p_map, [0.5001, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1], cmap=plt.cm.PuBu)
ax.contour(X, Y, p_map, [0.5001, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], colors='k', linestyles='--', linewidths=1)

# 添加概率分界文字
# line = ax.contour(X, Y, p_map, [0.5001, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], colors='k', linestyles='--', linewidths=1)
# ax.clabel(line, inline=True, fontsize=7)
# 保存或显示图片
# fig.savefig('v220824_img.svg')
plt.show()