from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

# base on https://alchemystars.fandom.com/wiki/Recruitment and publicity in the chinese server
__all__ = [
    'PITY_6STAR',
    'P_5',
    'P_4',
    'P_3',
    'P_6s',
    'P_5s',
    'P_4s',
    'P_3s',
    'common_6star',
    'common_5star',
    'common_4star',
    'common_3star',
    'up_6star',
]

# 白夜极光6星保底概率表
PITY_6STAR = np.zeros(91)
PITY_6STAR[1:51] = 0.02
PITY_6STAR[51:91] = np.arange(1, 41) * 0.025 + 0.02
PITY_6STAR[90] = 1
# 其他无保底物品初始概率
P_5 = 0.095
P_4 = 0.33
P_3 = 0.555
# 按照 stationary_p.py 的计算结果修订
P_6s = 0.02914069
P_5s = 0.095
P_4s = 0.33570489
P_3s = 0.54015442
# 定义获取星级物品的模型
common_6star = PityModel(PITY_6STAR)
common_5star = BernoulliGachaModel(P_5s)
common_4star = BernoulliGachaModel(P_4s)
common_3star = BernoulliGachaModel(P_3s)
# 定义获取UP物品模型
up_6star = DualPityModel(PITY_6STAR, [0, 0.5, 0.5, 1])

if __name__ == '__main__':
    print(common_6star(1)[90])
    print(up_6star(1)[270])
    pass