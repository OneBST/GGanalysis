from GGanalysis.games.girls_frontline2_exilium import *
from GGanalysis.markov import PriorityPityChain

# 调用预置工具计算在精英道具耦合普通道具情况下的概率，精英道具不会重置普通道具保底
gacha_system = PriorityPityChain([PITY_ELITE, PITY_COMMON], remove_pity=False)
print('精英道具及普通道具平稳概率', gacha_system.stationary_rates())
dist = gacha_system.interarrival_dist(item_type=1, max_steps=500)
print('普通道具相邻间隔前20抽', dist.dist[1:21], '超过500抽概率', dist.tail_mass)

# 调用预置工具计算武器池在精英道具耦合普通道具情况下的概率，精英道具不会重置普通道具保底
gacha_system = PriorityPityChain([PITY_ELITE_W, PITY_COMMON_W], remove_pity=False)
print('武器池精英道具及普通道具平稳概率', gacha_system.stationary_rates())
dist = gacha_system.interarrival_dist(item_type=1, max_steps=500)
print('武器池普通道具相邻间隔前20抽', dist.dist[1:21], '超过500抽概率', dist.tail_mass)
