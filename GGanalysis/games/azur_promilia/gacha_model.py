'''
    蓝色星原：旅谣（Azur Promilia）三测抽卡模型

    五星采用线性递增假设，将综合概率视为精确目标进行拟合。

    角色池
    五星：基础概率 0.8%，第71抽开始概率上升（1~70抽固定0.8%），
            第90抽硬保底，综合概率 1.58%，角色池不歪
    四星：基础概率 6%，10抽保底，四星中 50% 为UP角色、25%为其他四星角色、
            25%为四星灵子（武器），每个卡池UP两个四星，综合概率公示 12%
    武器池
    五星：基础概率 1.2%，第61抽开始概率上升（1~60抽固定1.2%），
            第80抽硬保底，综合概率 2.09%
    四星：基础概率 6%，综合概率公示 12%，四星中 50% 为UP四星灵子（武器）、
            25%为其他四星灵子（武器）、25%为四星角色
'''
import numpy as np
from GGanalysis.distribution_1d import linear_p_increase
from GGanalysis.basic_models import PityBernoulliModel, PityModel

__all__ = [
    'BASE_P', 'SOFT_PITY_BEGIN', 'HARD_PITY',
    'BASE_P_4STAR', 'HARD_PITY_4STAR',
    'PITY_5STAR',
    'PITY_4STAR',
    'common_5star',
    'up_5star_character',
    'common_4star',
    'up_4star_character',
    'up_4star_specific_character',
    'stander_4star_character',
    'four_star_weapon',
    'BASE_P_W5STAR',
    'SOFT_PITY_BEGIN_W5STAR', 'HARD_PITY_W5STAR',
    'STEP_5STAR', 'STEP_W5STAR', 'PITY_W5STAR',
    'common_5star_weapon', 'FOUR_STAR_CHARACTER_POOL', 'FOUR_STAR_WEAPON_POOL',
]

# 公示参数
BASE_P = 0.008              # 五星基础概率
SOFT_PITY_BEGIN = 71        # 概率开始上升的位置
HARD_PITY = 90              # 五星硬保底位置
BASE_P_4STAR = 0.06         # 四星基础概率
HARD_PITY_4STAR = 10        # 四星硬保底位置
BASE_P_W5STAR = 0.012
SOFT_PITY_BEGIN_W5STAR = 61
HARD_PITY_W5STAR = 80
# 条件于抽到四星的类别比例；不推断具体UP数量或大小保底。
FOUR_STAR_CHARACTER_POOL = {'up_character': 0.5, 'other_character': 0.25, 'weapon': 0.25}
FOUR_STAR_WEAPON_POOL = {'up_weapon': 0.5, 'other_weapon': 0.25, 'character': 0.25}


# 线性假设下的拟合结果；复算逻辑见 pity_inference.py，导入模型时不执行拟合。
STEP_5STAR = 0.0016723364963552529
STEP_W5STAR = 0.011901857189272795
PITY_5STAR = linear_p_increase(BASE_P, SOFT_PITY_BEGIN, STEP_5STAR, HARD_PITY)
PITY_5STAR[-1] = 1
PITY_W5STAR = linear_p_increase(
    BASE_P_W5STAR, SOFT_PITY_BEGIN_W5STAR, STEP_W5STAR, HARD_PITY_W5STAR)
PITY_W5STAR[-1] = 1
# 蓝色星原普通4星保底概率表：1~9抽6%，第10抽保底
PITY_4STAR = np.zeros(HARD_PITY_4STAR+1)
PITY_4STAR[1:HARD_PITY_4STAR] = BASE_P_4STAR
PITY_4STAR[HARD_PITY_4STAR] = 1

# 定义获取星级物品的模型
common_5star = PityModel(PITY_5STAR)
common_5star_weapon = PityModel(PITY_W5STAR)
common_4star = PityModel(PITY_4STAR)
# 定义蓝色星原角色池模型，卡池不歪，获得五星即为UP五星
up_5star_character = common_5star
# 四星构成：50%为UP角色，每个卡池UP两个故指定四星为25%，其余为25%其他四星角色与25%四星灵子
up_4star_character = PityBernoulliModel(PITY_4STAR, 0.5)
up_4star_specific_character = PityBernoulliModel(PITY_4STAR, 0.25)
stander_4star_character = PityBernoulliModel(PITY_4STAR, 0.25)
four_star_weapon = PityBernoulliModel(PITY_4STAR, 0.25)
