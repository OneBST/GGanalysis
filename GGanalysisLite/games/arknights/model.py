from GGanalysisLite.distribution_1d import *
from GGanalysisLite.gacha_layers import *
from GGanalysisLite.basic_models import *

# 获取普通六星
class AN_6star_common(common_gacha_model):
    def __init__(self) -> None:
        super().__init__()
        # 设置保底概率
        pity_p = np.zeros(100)
        pity_p[1:51] = 0.02
        pity_p[51:99] = np.arange(1, 49) * 0.02 + 0.02
        pity_p[99] = 1
        self.layers = [Pity_layer(pity_p)]
    
    def get_dist(self, item_num: int=1, pull_state: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = [[[], {'pull_state':pull_state}]]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取单UP六星
class AN_6star_singleup(AN_6star_common):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(Bernoulli_layer(1/2))

    def get_dist(self, item_num: int=1, pull_state: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = [[[], {'pull_state':pull_state}], [[], {}]]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)

# 获取双UP六星中的特定六星
class AN_6star_dualup(AN_6star_common):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(Bernoulli_layer(1/4))

    def get_dist(self, item_num: int=1, pull_state: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = [[[], {'pull_state':pull_state}], [[], {}]]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)
    
# 获取限定UP六星中的限定六星
class AN_6star_limitup(AN_6star_common):
    def __init__(self) -> None:
        super().__init__()
        self.layers.append(Bernoulli_layer(0.35))

    def get_dist(self, item_num: int=1, pull_state: int=0, multi_dist: bool=False) -> finite_dist_1D:
        parameter_list = [[[], {'pull_state':pull_state}], [[], {}]]
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        return self._get_dist(item_num, parameter_list)