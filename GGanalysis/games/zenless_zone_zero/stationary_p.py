from GGanalysis.games.zenless_zone_zero import PITY_5STAR, PITY_4STAR, PITY_W5STAR, PITY_W4STAR
from GGanalysis.markov import PriorityPityChain

# 调用预置工具计算在五星四星耦合情况下的概率
common_gacha_system = PriorityPityChain([PITY_5STAR, PITY_4STAR, [0, 1]], remove_pity=True)
print('常驻及角色池概率', common_gacha_system.stationary_rates())
dist = common_gacha_system.interarrival_dist(item_type=1, max_steps=500)
print('相邻两次获得四星的间隔分布前20抽', dist.dist[1:21], '超过500抽概率', dist.tail_mass)

weapon_gacha_system = PriorityPityChain([PITY_W5STAR, PITY_W4STAR, [0, 1]], remove_pity=True)
print('音擎及邦布池概率', weapon_gacha_system.stationary_rates())
dist = weapon_gacha_system.interarrival_dist(item_type=1, max_steps=500)
print('音擎及邦布池相邻两次获得四星的间隔分布前20抽', dist.dist[1:21], '超过500抽概率', dist.tail_mass)
