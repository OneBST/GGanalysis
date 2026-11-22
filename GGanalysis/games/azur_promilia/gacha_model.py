"""蓝色星原：旅谣公测前瞻（2026-11-20）抽卡模型。

角色池与灵子池的共通规则
    五星概率开始递增后，前十抽每抽加 2 个百分点，之后每抽加 8 个百分点，直至硬保底。
    限定五星不歪，同类限定池跨期继承保底。

角色池（限定：桃笺为契、胧月绮影等；常驻：恒常观测）
    五星：第 71 抽起递增，第 90 抽保底。
    常驻自选：首次累计 50 抽、此后每累计 200 抽可额外自选一次，不计入抽取模型。

灵子池（限定：绯影入怀、一刃入鞘；常驻：恒序观测）
    五星：第 61 抽起递增，第 80 抽保底。
    常驻定向：选定目标后，抽出的五星必为所选灵子。

新手池（始源观测，未建模）
    仅限十连，每十连消耗 8 个纯粹忆痕。
    保底独立计算，30 抽内必出五星角色，满 30 抽关闭。
"""
import numpy as np
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

STEP_5STAR = STEP_W5STAR = 0.02
_LATE_STEP = 0.08

PITY_5STAR = np.zeros(HARD_PITY + 1)
PITY_5STAR[1:71] = BASE_P
PITY_5STAR[71:81] = BASE_P + np.arange(1, 11) * STEP_5STAR
PITY_5STAR[81:90] = PITY_5STAR[80] + np.arange(1, 10) * _LATE_STEP
PITY_5STAR[90] = 1

PITY_W5STAR = np.zeros(HARD_PITY_W5STAR + 1)
PITY_W5STAR[1:61] = BASE_P_W5STAR
PITY_W5STAR[61:71] = BASE_P_W5STAR + np.arange(1, 11) * STEP_W5STAR
PITY_W5STAR[71:80] = PITY_W5STAR[70] + np.arange(1, 10) * _LATE_STEP
PITY_W5STAR[80] = 1
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
