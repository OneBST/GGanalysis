'''
    蓝色星原：旅谣（Azur Promilia）抽卡模型

    官方（旅迹测试）公示信息：
        五星：基础概率 0.8%，第71抽开始概率上升（1~70抽固定0.8%），
              第90抽硬保底，综合概率 1.58%，角色池不歪
        四星：基础概率 6%，10抽保底，四星中 50% 为UP角色、25%为其他四星角色、
              25%为四星灵子（武器），每个卡池UP两个四星，综合概率公示 12%

    五星概率上升模型尚未由实测确认。公示的综合概率 1.58% 反过来限制了曲线陡度：
    原神式「第71抽起线性递增、第90抽恰好升至100%」会得到 1.7633%，与公示矛盾；
    71抽起完全不设软保底则只有 1.5544%。故软保底只能极平缓，本模块取
    第71抽起每抽 +0.167% 的线性模型，综合概率 1.579967%。

    四星部分未纳入「五星挤四星」的交互，与原神模块口径一致，会略微高估
    （近似 13.0043%，精确 12.9366%，公示值 12%），详见 pity_inference.py。
'''
from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

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
]

# 公示参数
BASE_P = 0.008              # 五星基础概率
SOFT_PITY_BEGIN = 71        # 概率开始上升的位置
HARD_PITY = 90              # 五星硬保底位置
BASE_P_4STAR = 0.06         # 四星基础概率
HARD_PITY_4STAR = 10        # 四星硬保底位置

# 蓝色星原普通5星保底概率表：1~70抽0.8%，第71抽起每抽+0.167%，第90抽保底
PITY_5STAR = np.zeros(91)
PITY_5STAR[1:SOFT_PITY_BEGIN] = BASE_P
PITY_5STAR[SOFT_PITY_BEGIN:HARD_PITY] = np.arange(1, HARD_PITY-SOFT_PITY_BEGIN+1) * 0.00167 + BASE_P
PITY_5STAR[HARD_PITY] = 1
# 蓝色星原普通4星保底概率表：1~9抽6%，第10抽保底
PITY_4STAR = np.zeros(HARD_PITY_4STAR+1)
PITY_4STAR[1:HARD_PITY_4STAR] = BASE_P_4STAR
PITY_4STAR[HARD_PITY_4STAR] = 1

# 定义获取星级物品的模型
common_5star = PityModel(PITY_5STAR)
common_4star = PityModel(PITY_4STAR)
# 定义蓝色星原角色池模型，卡池不歪，获得五星即为UP五星
up_5star_character = common_5star
# 四星构成：50%为UP角色，每个卡池UP两个故指定四星为25%，其余为25%其他四星角色与25%四星灵子
up_4star_character = PityBernoulliModel(PITY_4STAR, 0.5)
up_4star_specific_character = PityBernoulliModel(PITY_4STAR, 0.25)
stander_4star_character = PityBernoulliModel(PITY_4STAR, 0.25)
four_star_weapon = PityBernoulliModel(PITY_4STAR, 0.25)

if __name__ == '__main__':
    print("五星综合概率", 1/p2exp(PITY_5STAR))
    print("四星综合概率", 1/p2exp(PITY_4STAR))
    print("UP五星角色期望", up_5star_character(1).exp)
    print("UP四星角色期望", up_4star_character(1).exp)
    print("指定UP四星角色期望", up_4star_specific_character(1).exp)
