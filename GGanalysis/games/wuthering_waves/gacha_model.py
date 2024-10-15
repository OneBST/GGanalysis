'''
    注意，本模块使用概率模型仅为根据游戏测试阶段推测，不能保证完全准确
    分析文章 https://www.bilibili.com/read/cv34870533/
'''
from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

__all__ = [
    'PITY_5STAR',
    'PITY_4STAR',
    'common_5star',
    'common_4star',
    'up_5star_character',
    'up_4star_character',
    'up_4star_specific_character',
    'up_5star_weapon',
    'up_4star_weapon',
    'up_4star_specific_weapon',
]

# 鸣潮普通5星保底概率表
PITY_5STAR = np.zeros(80)
PITY_5STAR[1:66] = 0.008
PITY_5STAR[66:71] = np.arange(1, 5+1) * 0.04 + PITY_5STAR[65]
PITY_5STAR[71:76] = np.arange(1, 5+1) * 0.08 + PITY_5STAR[70]
PITY_5STAR[76:80] = np.arange(1, 4+1) * 0.1 + PITY_5STAR[75]
PITY_5STAR[79] = 1
# 鸣潮普通4星保底概率表
PITY_4STAR = np.zeros(11)
PITY_4STAR[1:10] = 0.06
PITY_4STAR[10] = 1

# 定义获取星级物品的模型
common_5star = PityModel(PITY_5STAR)
common_4star = PityModel(PITY_4STAR)
# 定义鸣潮角色池模型
up_5star_character = DualPityModel(PITY_5STAR, [0, 0.5, 1])
up_4star_character = DualPityModel(PITY_4STAR, [0, 0.5, 1])
up_4star_specific_character = DualPityBernoulliModel(PITY_4STAR, [0, 0.5, 1], 1/3)
# 定义鸣潮武器池模型
up_5star_weapon = PityModel(PITY_5STAR)
up_4star_weapon = DualPityModel(PITY_4STAR, [0, 0.5, 1])
up_4star_specific_weapon = DualPityBernoulliModel(PITY_4STAR, [0, 0.5, 1], 1/3)

if __name__ == '__main__':
    print(1.5*common_5star(1).exp)
    print(PITY_5STAR[70])
    print(PITY_5STAR[76])
    '''
    print(PITY_5STAR[70:])
    close_dis = 1
    pity_begin = 0
    p_raise = 0
    for i in range(50, 75+1):
        # 枚举开始上升位置
        PITY_5STAR = np.zeros(81)
        PITY_5STAR[1:i] = 0.008
        for j in range(1, 10):
            # 枚举每抽上升概率
            p_step = j / 100
            PITY_5STAR[i:80] = np.arange(1, 80-i+1) * p_step + 0.008
            PITY_5STAR[80] = 1
            common_5star = PityModel(PITY_5STAR)
            p = 1 / common_5star(1).exp
            if p > 0.018:
                # 达到要求进行记录
                if p-0.018 < close_dis:
                    close_dis = p-0.018
                    pity_begin = i
                    p_raise = p_step
                    print(p, i, p_step, PITY_5STAR[70:81])
    '''
    