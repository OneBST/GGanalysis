from GGanalysis.games.wuthering_waves import PITY_5STAR, PITY_4STAR
from GGanalysis.markov import PriorityPityChain

# 调用预置工具计算在1.0版本之后五星四星耦合情况下的概率
common_gacha_system = PriorityPityChain([PITY_5STAR, PITY_4STAR, [0, 1]], remove_pity=True)
print('卡池各星级期望抽数', 1/common_gacha_system.stationary_rates())
print('卡池各星级综合概率', common_gacha_system.stationary_rates())
dist = common_gacha_system.interarrival_dist(item_type=1, max_steps=500)
print('相邻两次获得四星的间隔分布前20抽', dist.dist[1:21], '超过500抽概率', dist.tail_mass)
