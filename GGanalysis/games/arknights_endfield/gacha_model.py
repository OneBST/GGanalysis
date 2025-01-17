from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

'''
    注意，本模块按明日方舟：终末地二测公示进行简单建模，非公测版本
'''

__all__ = [
    'PITY_6STAR',
    'common_6star',
    'up_6star_character',
]

# 设置6星概率递增表
PITY_6STAR = np.zeros(81)
PITY_6STAR[1:65+1] = 0.008
PITY_6STAR[66:81] = np.arange(1, 15+1) * 0.05 + 0.008
PITY_6STAR[80] = 1

# ★★★★★★
common_6star = PityModel(PITY_6STAR)
up_6star_character = PityBernoulliModel(PITY_6STAR, 0.5)  # 不考虑第一个

# 五星公示基础概率为8%，更多信息还有待发掘

from matplotlib import pyplot as plt
if __name__ == '__main__':
    pass
