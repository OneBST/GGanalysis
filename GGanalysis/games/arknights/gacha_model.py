from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

'''
    注意，本模块按公示概率进行建模，但忽略了一些情况
        如不纳入对300保底的考虑，获取1个物品的分布不会在第300抽处截断
        这么做的原因是，模型只支持一抽最多获取1个物品，若在第300抽处刚好抽到物品
        等于一抽抽到两个物品，无法处理。
        对于这个问题，建议结合抽数再加一步分析，进行一次后处理（如使用 QuantileFunction 中的 direct_exchange 项）
'''

__all__ = [
    'PITY_6STAR',
    'PITY_5STAR',
    'P_5STAR_AVG',
    'common_6star',
    'single_up_6star',
    'dual_up_specific_6star',
    'limited_up_6star',
    'common_5star',
    'single_up_specific_5star',
    'dual_up_specific_5star',
    'triple_up_specific_5star',
]

# 设置六星概率递增表
PITY_6STAR = np.zeros(100)
PITY_6STAR[1:51] = 0.02
PITY_6STAR[51:99] = np.arange(1, 49) * 0.02 + 0.02
PITY_6STAR[99] = 1
# 设置五星概率递增表（五星保底会被六星挤掉，所以需要做一点近似）
PITY_5STAR = np.zeros(42, dtype=float)
PITY_5STAR[:16] = 0.08
PITY_5STAR[16:21] = np.arange(1, 6) * 0.02 + 0.08
PITY_5STAR[21:41] = np.arange(1, 21) * 0.04 + 0.18
PITY_5STAR[41] = 1
# 设置五星综合概率，用于近似计算
P_5STAR_AVG = 0.08948

# 获取普通六星
common_6star = PityModel(PITY_6STAR)
# 获取单UP六星
single_up_6star = PityBernoulliModel(PITY_6STAR, 1/2)
# 获取双UP六星中的特定六星
dual_up_specific_6star = PityBernoulliModel(PITY_6STAR, 1/4)    
# 获取限定UP六星中的限定六星
limited_up_6star = PityBernoulliModel(PITY_6STAR, 0.35)

# 方舟限定池同时获取限定六星及陪跑六星
class AK_Limit_Model(CommonGachaModel):
    def __init__(self, pity_p1, p, collect_item=2, e_error = 1e-8, max_dist_len=1e5) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p1))
        self.layers.append(BernoulliLayer(p, e_error, max_dist_len))
        self.layers.append(CouponCollectorLayer(2, collect_item))

    def __call__(self, item_num: int = 1, multi_dist: bool = False, pull_state = 0, up_guarantee = 0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, pull_state, up_guarantee, *args, **kwds)

    def _build_parameter_list(self, pull_state: int=0, up_guarantee: int=0) -> list:
        parameter_list = [
            [[], {'pull_state':pull_state}],
            [[], {}],
            [[], {}]
        ]
        return parameter_list
# 同时获取两个UP六星
both_up_6star = AK_Limit_Model(PITY_6STAR, 0.7, collect_item=2)

# 五星公示概率为的8%，实际上综合概率为8.948% 这里按照综合概率近似计算
# 获取普通五星
common_5star = BernoulliGachaModel(P_5STAR_AVG)
# 获取单UP五星
single_up_specific_5star = BernoulliGachaModel(P_5STAR_AVG/2)
# 获取双UP五星中的特定五星
dual_up_specific_5star = BernoulliGachaModel(P_5STAR_AVG/2/2)
# 获取三UP五星中的特定五星
triple_up_specific_5star = BernoulliGachaModel(P_5STAR_AVG/2/3)

