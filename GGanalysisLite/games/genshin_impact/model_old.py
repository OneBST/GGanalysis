from GGanalysisLite.distribution_1d import *
from GGanalysisLite.gacha_layers import *
from GGanalysisLite.basic_models import *

# 获取普通五星
class GI_5star_common(common_gacha_model):
    def __init__(self) -> None:
        super().__init__()
        # 设置保底概率
        pity_p = np.zeros(91)
        pity_p[1:74] = 0.006
        pity_p[74:90] = np.arange(1, 17) * 0.06 + 0.006
        pity_p[90] = 1
        self.layers = [Pity_layer(pity_p)]
    
    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        parameter_list = [[[], {'pull_state':kwds['pull_state']}]]
        return parameter_list

    def get_dist(self, item_num: int=1, pull_state: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(pull_state=pull_state)  # [[[], {'pull_state':pull_state}]]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取UP五星角色
class GI_5star_upcharacter(GI_5star_common):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(Pity_layer([0, 0.5, 1]))

    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        l1_param = [[], {'pull_state':kwds['pull_state']}]
        l2_param = [[], {'pull_state':kwds['up_guarantee']}]
        parameter_list = [l1_param, l2_param]
        return parameter_list

    def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(pull_state=pull_state, up_guarantee=up_guarantee)
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取普通四星
class GI_4star_common(common_gacha_model):
    def __init__(self) -> None:
        super().__init__()
        # 设置保底概率
        pity_p = np.zeros(11)
        pity_p[1:9] = 0.051
        pity_p[9] = 0.051 + 0.51
        pity_p[10] = 1
        self.layers = [Pity_layer(pity_p)]

    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        parameter_list = [[[], {'pull_state':kwds['pull_state']}]]
        return parameter_list

    def get_dist(self, item_num: int=1, pull_state: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(pull_state=pull_state)
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取UP四星角色
class GI_4star_upcharacter(GI_4star_common):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(Pity_layer([0, 0.5, 1]))

    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        l1_param = [[], {'pull_state':kwds['pull_state']}]
        l2_param = [[], {'pull_state':kwds['up_guarantee']}]
        parameter_list = [l1_param, l2_param]
        return parameter_list

    def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(pull_state=pull_state, up_guarantee=up_guarantee)
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取特定UP四星角色
class GI_4star_specific_upcharacter(GI_4star_upcharacter):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(Bernoulli_layer(1/3))

    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        l1_param = [[], {'pull_state':kwds['pull_state']}]
        l2_param = [[], {'pull_state':kwds['up_guarantee']}]
        l3_param = [[], {}]
        parameter_list = [l1_param, l2_param, l3_param]
        return parameter_list

    def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(pull_state=pull_state, up_guarantee=up_guarantee)
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取普通五星武器
class GI_5star_weapon(common_gacha_model):
    def __init__(self) -> None:
        super().__init__()
        # 设置保底概率
        pity_p = np.zeros(78)
        pity_p[1:63] = 0.007
        pity_p[63:77] = np.arange(1, 15) * 0.07 + 0.007
        pity_p[77] = 1
        self.layers = [Pity_layer(pity_p)]
    
    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        parameter_list = [[[], {'pull_state':kwds['pull_state']}]]
        return parameter_list

    def get_dist(self, item_num: int=1, pull_state: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(pull_state=pull_state)
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取UP五星武器
class GI_5star_upweapon(GI_5star_weapon):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(Pity_layer([0, 0.75, 1]))

    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        l1_param = [[], {'pull_state':kwds['pull_state']}]
        l2_param = [[], {'pull_state':kwds['up_guarantee']}]
        parameter_list = [l1_param, l2_param]
        return parameter_list

    def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(pull_state=pull_state, up_guarantee=up_guarantee)
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 定轨获取特定UP五星武器
class GI_5star_upweapon_EP(GI_5star_weapon):
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

    def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, fate_point: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(pull_state=pull_state, up_guarantee=up_guarantee, fate_point=fate_point)
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取武器池四星
class GI_4star_weapon(common_gacha_model):
    def __init__(self) -> None:
        super().__init__()
        # 设置保底概率
        pity_p = np.zeros(10)
        pity_p[1:8] = 0.06
        pity_p[8] = 0.06 + 0.6
        pity_p[9] = 1
        self.layers = [Pity_layer(pity_p)]
    
    def get_dist(self, item_num: int=1, pull_state: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = [[[], {'pull_state':pull_state}]]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取UP四星武器
class GI_4star_upweapon(GI_4star_weapon):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(Pity_layer([0, 0.75, 1]))

    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        l1_param = [[], {'pull_state':kwds['pull_state']}]
        l2_param = [[], {'pull_state':kwds['up_guarantee']}]
        parameter_list = [l1_param, l2_param]
        return parameter_list

    def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(pull_state=pull_state, up_guarantee=up_guarantee)
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取特定UP四星武器
class GI_4star_specific_upweapon(GI_4star_upweapon):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(Bernoulli_layer(1/5))

    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        l1_param = [[], {'pull_state':kwds['pull_state']}]
        l2_param = [[], {'pull_state':kwds['up_guarantee']}]
        l3_param = [[], {}]
        parameter_list = [l1_param, l2_param, l3_param]
        return parameter_list

    def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(pull_state=pull_state, up_guarantee=up_guarantee)
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)