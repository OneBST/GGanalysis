from GGanalysis.games.wuthering_waves import PITY_5STAR, PITY_4STAR
from GGanalysis.markov_method import PriorityPitySystem

# 调用预置工具计算在1.0版本之后五星四星耦合情况下的概率
common_gacha_system = PriorityPitySystem([PITY_5STAR, PITY_4STAR, [0, 1]], remove_pity=True)
print('卡池各星级综合概率', 1/common_gacha_system.get_stationary_p())
print('四星非10抽保底概率（考虑重置保底）', common_gacha_system.get_type_distribution(type=1))