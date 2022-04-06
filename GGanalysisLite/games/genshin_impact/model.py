from GGanalysisLite.distribution_1d import *
from GGanalysisLite.gacha_layers import *
from GGanalysisLite.basic_models import *

'''
    注意，本模块对四星概率进行了近似处理
        1. 仅考虑不存在五星物品时的情况，没有考虑四星物品被五星物品挤到下一抽的可能
        2. 对于UP四星物品，没有考虑UP四星从常驻中也有概率获取的可能性
    计算所得四星综合概率会略高于实际值，获取UP四星的概率略低于实际值，但影响非常微弱可以忽略
    
    同时由于复杂性，原神的平稳机制没有纳入计算，其影响也很低。如果想了解平稳机制的影响，可以使用 GGanalysislib 工具包
    见 https://github.com/OneBST/GGanalysis
'''

# 原神普通五星保底概率表
pity_5_star = np.zeros(91)
pity_5_star[1:74] = 0.006
pity_5_star[74:90] = np.arange(1, 17) * 0.06 + 0.006
pity_5_star[90] = 1
# 原神普通四星保底概率表
pity_4_star = np.zeros(11)
pity_4_star[1:9] = 0.051
pity_4_star[9] = 0.051 + 0.51
pity_4_star[10] = 1
# 原神武器池五星保底概率表
pity_w5_star = np.zeros(78)
pity_w5_star[1:63] = 0.007
pity_w5_star[63:77] = np.arange(1, 15) * 0.07 + 0.007
pity_w5_star[77] = 1
# 原神武器池四星保底概率表
pity_w4_star = np.zeros(10)
pity_w4_star[1:8] = 0.06
pity_w4_star[8] = 0.06 + 0.6
pity_w4_star[9] = 1

# 定义获取星级物品的模型
Common_5star = pity_model(pity_5_star)
Common_4star = pity_model(pity_4_star)
# 定义原神角色池模型
UP_5star_character = dual_pity_model(pity_5_star, [0, 0.5, 1])
UP_4star_character = dual_pity_model(pity_4_star, [0, 0.5, 1])
UP_4star_specific_character = dual_pity_bernoulli_model(pity_4_star, [0, 0.5, 1], 1/3)
# 定义原神武器池模型
Common_5star_weapon = pity_model(pity_w5_star)
Common_4star_weapon = pity_model(pity_w4_star)
UP_5star_weapon = dual_pity_model(pity_w5_star, [0, 0.75, 1])
UP_4star_weapon = dual_pity_model(pity_w4_star, [0, 0.75, 1])
UP_4star_specific_weapon = dual_pity_bernoulli_model(pity_w4_star, [0, 0.75, 1], 1/5)

# 定轨获取特定UP五星武器
class genshin_5star_EP_weapon(common_gacha_model):
    def __init__(self) -> None:
        super().__init__()
        self.state_num = {'get':0, 'fate0pity':1, 'fate1':2, 'fate1pity':3, 'fate2':4}
        state_trans = [
            ['get', 'get', 0.375],
            ['get', 'fate1', 0.375],
            ['get', 'fate1pity', 0.25],
            ['fate0pity', 'get', 0.5],
            ['fate0pity', 'fate1', 0.5],
            ['fate1', 'get', 0.375],
            ['fate1', 'fate2', 1-0.375],
            ['fate1pity', 'get', 0.5],
            ['fate1pity', 'fate2', 0.5],
            ['fate2', 'get', 1]
        ]
        M = table2matrix(self.state_num, state_trans)
        self.layers.append(Pity_layer(pity_w5_star))
        self.layers.append(Markov_layer(M))

    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        fate_point = kwds['fate_point']
        up_guarantee = kwds['up_guarantee']
        if fate_point >= 2:
            begin_pos = 4
        elif fate_point == 1 and up_guarantee == 1:
            begin_pos = 3
        elif fate_point == 1 and up_guarantee == 0:
            begin_pos = 2
        elif fate_point == 0 and up_guarantee == 1:
            begin_pos = 1
        else:
            begin_pos = 0
        l1_param = [[], {'pull_state':kwds['pull_state']}]
        l2_param = [[], {'begin_pos':begin_pos}]
        parameter_list = [l1_param, l2_param]
        return parameter_list
    # def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, fate_point: int=0, multi_dist: bool=False) -> finite_dist_1D:
    #     parameter_list = self._build_parameter_list(pull_state=pull_state, up_guarantee=up_guarantee, fate_point=fate_point)
    #     if multi_dist:
    #         return self._get_multi_dist(item_num, parameter_list)
    #     return self._get_dist(item_num, parameter_list)

UP_5star_EP_weapon = genshin_5star_EP_weapon()