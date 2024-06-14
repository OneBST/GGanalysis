'''
    注意，本模块使用概率模型仅为根据游戏测试阶段推测，不能保证完全准确
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
    'weapon_5star',
    'weapon_4star',
    'up_5star_character',
    'up_4star_character',
    'up_4star_specific_character',
    'up_5star_weapon',
    'up_4star_weapon',
    'up_4star_specific_weapon',
]

# 绝区零普通5星保底概率表
PITY_5STAR = np.zeros(91)
PITY_5STAR[1:74] = 0.006
PITY_5STAR[74:90] = np.arange(1, 17) * 0.06 + 0.006
PITY_5STAR[90] = 1
# 绝区零普通4星保底概率表 基础概率9.4% 角色池其中角色为7.05% 音擎为2.35% 十抽保底 综合14.4%
# 角色池暂时理解为没有UP机制介入时角色占比 3/4 音擎占比 1/4 
PITY_4STAR = np.zeros(11)
PITY_4STAR[1:10] = 0.094
PITY_4STAR[10] = 1

# 绝区零音擎5星保底概率表 基础概率1% 综合概率2% 80保底 75%概率单UP
PITY_W5STAR = np.zeros(81)
PITY_W5STAR[1:64] = 0.01
PITY_W5STAR[61:80] = np.arange(1, 20) * 0.05 + 0.01
PITY_W5STAR[80] = 1
# 绝区零音擎4星保底概率表 基础概率15% 其中音擎占13.125% 角色占1.875% 10抽保底 综合概率18% 75%概率UP
# 音擎池暂时理解为没有UP机制介入时音擎占比 7/8 角色占比 1/8
PITY_W4STAR = np.zeros(11)
PITY_W4STAR[1:10] = 0.15
PITY_W4STAR[10] = 1

# 定义获取星级物品的模型
common_5star = PityModel(PITY_5STAR)
common_4star = PityModel(PITY_4STAR)
weapon_5star = PityModel(PITY_W5STAR)
weapon_4star = PityModel(PITY_W4STAR)
# 定义绝区零角色池模型
up_5star_character = DualPityModel(PITY_5STAR, [0, 0.5, 1])
up_4star_character = DualPityModel(PITY_4STAR, [0, 0.5, 1])
up_4star_specific_character = DualPityBernoulliModel(PITY_4STAR, [0, 0.5, 1], 1/2)
# 定义绝区零武器池模型
up_5star_weapon = DualPityModel(PITY_W5STAR, [0, 0.75, 1])
up_4star_weapon = DualPityModel(PITY_W4STAR, [0, 0.75, 1])
up_4star_specific_weapon = DualPityBernoulliModel(PITY_W4STAR, [0, 0.75, 1], 1/2)
# 定义绝区零邦布池模型
# 暂时没写

if __name__ == '__main__':
    print(1/common_5star(1).exp)
    print(1/weapon_5star(1).exp)
    print(1/common_4star(1).exp)
    print(1/weapon_4star(1).exp)

    '''
    close_dis = 1
    pity_begin = 0
    p_raise = 0
    for i in range(60, 75+1):
        # 枚举开始上升位置
        PITY_5STAR = np.zeros(81)
        PITY_5STAR[1:i] = 0.01
        for j in range(5, 10):
            # 枚举每抽上升概率
            p_step = j / 100
            PITY_5STAR[i:80] = np.arange(1, 80-i+1) * p_step + 0.01
            PITY_5STAR[80] = 1
            common_5star = PityModel(PITY_5STAR)
            p = 1 / common_5star(1).exp
            if p > 0.02:
                # 达到要求进行记录
                if p-0.02 < close_dis:
                    close_dis = p-0.02
                    pity_begin = i
                    p_raise = p_step
                    print(p, i, p_step, PITY_5STAR[70:81])
    '''
    