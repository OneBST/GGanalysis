from GGanalysis.games.girls_frontline2_exilium import *
from GGanalysis.stationary_distribution_method import PriorityPitySystem

# 调用预置工具计算在精英道具耦合普通道具情况下的概率，精英道具不会重置普通道具保底
gacha_system = PriorityPitySystem([PITY_ELITE, PITY_COMMON], remove_pity=False)
print('精英道具及普通道具平稳概率', gacha_system.get_stationary_p())
print('普通道具的抽数分布', gacha_system.get_type_distribution(type=1))

# 调用预置工具计算武器池在精英道具耦合普通道具情况下的概率，精英道具不会重置普通道具保底
gacha_system = PriorityPitySystem([PITY_ELITE_W, PITY_COMMON_W], remove_pity=False)
print('武器池精英道具及普通道具平稳概率', gacha_system.get_stationary_p())
print('武器池普通道具的抽数分布', gacha_system.get_type_distribution(type=1))