from GGanalysisLite.distribution_1d import *
from GGanalysisLite.gacha_layers import *
from GGanalysisLite.basic_models import *

class GI_5star_common(common_gacha_model):
    def __init__(self) -> None:
        super().__init__()
        # 设置保底概率
        pity_p = np.zeros(91)
        pity_p[1:74] = 0.006
        pity_p[74:90] = np.arange(1, 17) * 0.06 + 0.006
        pity_p[90] = 1
        self.layers = [pity_layer(pity_p)]
        # print(pity_p)
    
    def get_dist(self, item_num: int=1, pull_state: int=0, multi_dist: bool=False):
        parameter_list = [[[], {'pull_state':pull_state}]]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)


class GI_5star_upcharacter(GI_5star_common):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(pity_layer([0, 0.5, 1]))

    def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, multi_dist: bool=False):
        l1_param = [[], {'pull_state':pull_state}]
        l2_param = [[], {'pull_state':up_guarantee}]
        parameter_list = [l1_param, l2_param]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

class GI_4star_common(common_gacha_model):
    def __init__(self) -> None:
        super().__init__()
        # 设置保底概率
        pity_p = np.zeros(11)
        pity_p[1:9] = 0.051
        pity_p[9] = 0.051 + 0.51
        pity_p[10] = 1
        self.layers = [pity_layer(pity_p)]

    def get_dist(self, item_num: int=1, pull_state: int=0, multi_dist: bool=False):
        parameter_list = [[[], {'pull_state':pull_state}]]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

class GI_4star_upcharacter(GI_4star_common):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(pity_layer([0, 0.5, 1]))

    def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, multi_dist: bool=False):
        l1_param = [[], {'pull_state':pull_state}]
        l2_param = [[], {'pull_state':up_guarantee}]
        parameter_list = [l1_param, l2_param]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

class GI_4star_specific_upcharacter(GI_4star_upcharacter):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(bernoulli_layer(1/3))

    def get_dist(self, item_num: int=1, pull_state: int=0, up_guarantee: int=0, multi_dist: bool=False):
        l1_param = [[], {'pull_state':pull_state}]
        l2_param = [[], {'pull_state':up_guarantee}]
        l3_param = [[], {}]
        parameter_list = [l1_param, l2_param, l3_param]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)