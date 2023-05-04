'''
    注意，本模块使用概率模型仅为推测，不能保证完全准确
'''
from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

__all__ = [
    'PITY_5STAR',
    'PITY_4STAR',
    'PITY_W5STAR',
    'PITY_W4STAR',
    'common_5star',
    'common_4star',
    'up_5star_character',
    'up_4star_character',
    'up_4star_specific_character',
    'common_5star_weapon',
    'common_4star_weapon',
    'up_5star_weapon',
    'up_4star_weapon',
    'up_4star_specific_weapon',
]

# 星穹铁道普通5星保底概率表
PITY_5STAR = np.zeros(91)
PITY_5STAR[1:74] = 0.006
PITY_5STAR[74:90] = np.arange(1, 17) * 0.06 + 0.006
PITY_5STAR[90] = 1
# 星穹铁道普通4星保底概率表
PITY_4STAR = np.zeros(11)
PITY_4STAR[1:9] = 0.051
PITY_4STAR[9] = 0.051 + 0.51
PITY_4STAR[10] = 1
# 星穹铁道武器池5星保底概率表
PITY_W5STAR = np.zeros(81)
PITY_W5STAR[1:67] = 0.008
PITY_W5STAR[67:71] = np.arange(1, 5) * 0.008 * 14 + 0.008
PITY_W5STAR[71:80] = np.arange(1, 10) * 0.008 * 7 + PITY_W5STAR[70]
PITY_W5STAR[80] = 1
# 星穹铁道武器池4星保底概率表
PITY_W4STAR = np.array([0,0.066,0.066,0.066,0.066,0.066,0.066,0.066,0.066+0.066*6,0.87,1])

# 定义获取星级物品的模型
common_5star = PityModel(PITY_5STAR)
common_4star = PityModel(PITY_4STAR)
# 定义星穹铁道角色池模型
up_5star_character = DualPityModel(PITY_5STAR, [0, 0.5, 1])
up_4star_character = DualPityModel(PITY_4STAR, [0, 0.5, 1])
up_4star_specific_character = DualPityBernoulliModel(PITY_4STAR, [0, 0.5, 1], 1/3)
# 定义星穹铁道武器池模型
common_5star_weapon = PityModel(PITY_W5STAR)
common_4star_weapon = PityModel(PITY_W4STAR)
up_5star_weapon = DualPityModel(PITY_W5STAR, [0, 0.75, 1])
up_4star_weapon = DualPityModel(PITY_W4STAR, [0, 0.75, 1])
up_4star_specific_weapon = DualPityBernoulliModel(PITY_W4STAR, [0, 0.75, 1], 1/3)


if __name__ == '__main__':
    print(PITY_W5STAR)