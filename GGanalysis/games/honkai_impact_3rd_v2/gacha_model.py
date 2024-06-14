from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

# 这里是没有什么东西的荒原喵~ 模型没做好
__all__ = [
]

# 崩坏3第二部扩充补给S级保底概率表（推测值）
PITY_S = np.zeros(91)
PITY_S[1:90] = 0.0072
PITY_S[90] = 1

# 定义获取各等级物品的模型
s_character = PityModel(PITY_S)


if __name__ == '__main__':
    print(s_character(1).exp, 1/s_character(1).exp)
    pass