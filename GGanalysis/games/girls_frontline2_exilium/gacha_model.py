from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

__all__ = [
    'PITY_ELITE',
    'PITY_COMMON',
    'PITY_ELITE_W',
    'PITY_COMMON_W',
    'common_elite',
    'common_common',
    'weapon_elite',
    'weapon_common',
    'up_elite',
    'up_common_character',
    'up_common_specific_character',
    'up_elite_weapon',
    'up_common_weapon',
    'up_common_specific_weapon',
]

# 少女前线2：追放普通精英道具保底概率表（推测值）
PITY_ELITE = np.zeros(81)
PITY_ELITE[1:59] = 0.006
PITY_ELITE[59:81] = np.arange(1, 23) * 0.047 + 0.006
PITY_ELITE[80] = 1
# 少女前线2：追放普通标准道具保底概率表（推测值）
PITY_COMMON = np.zeros(11)
PITY_COMMON[1:10] = 0.06
PITY_COMMON[10] = 1
# 少女前线2：追放普通旧式武器概率
P_OLD = 0.934

# 少女前线2：追放普通精英道具保底概率表（推测值）
PITY_ELITE_W = np.zeros(71)
PITY_ELITE_W[1:51] = 0.007
PITY_ELITE_W[51:71] = np.arange(1, 21) * 0.053 + 0.007
PITY_ELITE_W[70] = 1
# 少女前线2：追放普通标准道具保底概率表（推测值）
PITY_COMMON_W = np.zeros(11)
PITY_COMMON_W[1:10] = 0.07
PITY_COMMON_W[10] = 1


# 定义获取星级物品的模型
common_elite = PityModel(PITY_ELITE)
common_common = PityModel(PITY_COMMON)
weapon_elite = PityModel(PITY_ELITE_W)
weapon_common = PityModel(PITY_COMMON_W)


# 定义获取UP物品模型，以下为简单推测模型
up_elite = DualPityModel(PITY_ELITE, [0, 0.5, 1])
up_common_character = PityBernoulliModel(PITY_COMMON, p=0.25)
up_common_specific_character = PityBernoulliModel(PITY_COMMON, p=0.25/2)
up_elite_weapon = DualPityModel(PITY_ELITE_W, [0, 0.75, 1])
up_common_weapon = DualPityModel(PITY_COMMON_W, [0, 0.75, 1])
up_common_specific_weapon = DualPityBernoulliModel(PITY_COMMON_W, [0, 0.75, 1], 1/3)

if __name__ == '__main__':
    # print(PITY_ELITE[58:])
    # print(common_elite(1).exp, 1/common_elite(1).exp)
    print(up_elite(1).exp, 1/up_elite(1).exp)
    # print(PITY_COMMON)
    # print(common_common(1).exp, 1/common_common(1).exp)
    # print(PITY_ELITE[50:])
    # print(weapon_elite(1).exp, 1/weapon_elite(1).exp)
    print(up_elite_weapon(1).exp, 1/up_elite_weapon(1).exp)
    # print(PITY_COMMON_W)
    # print(weapon_common(1).exp, 1/weapon_common(1).exp)
    print(up_common_specific_character(1).exp, 1/up_common_specific_character(1).exp)
    pass