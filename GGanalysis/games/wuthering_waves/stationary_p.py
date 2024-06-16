from GGanalysis.games.wuthering_waves import PITY_5STAR, PITY_4STAR
from GGanalysis.stationary_distribution_method import PriorityPitySystem

# 调用预置工具计算在五星四星耦合情况下的概率
common_gacha_system = PriorityPitySystem([PITY_5STAR, PITY_4STAR, [0, 1]], remove_pity=True)
print('常驻及角色池概率', common_gacha_system.get_stationary_p())
print('四星非10抽保底概率（考虑重置保底）', common_gacha_system.get_type_distribution(type=1))
print('四星非10抽保底概率（考虑重置保底）', sum(common_gacha_system.get_type_distribution(type=1)[:10]))