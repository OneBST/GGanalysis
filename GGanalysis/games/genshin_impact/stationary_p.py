from GGanalysis.games.genshin_impact import PITY_5STAR, PITY_4STAR, PITY_W5STAR, PITY_W4STAR
from GGanalysis.markov import PriorityPityChain

# 调用预置工具计算在五星四星耦合情况下的概率
common_gacha_system = PriorityPityChain([PITY_5STAR, PITY_4STAR, [0, 1]])
print('常驻及角色池概率', common_gacha_system.stationary_rates())
weapon_gacha_system = PriorityPityChain([PITY_W5STAR, PITY_W4STAR, [0, 1]])
print('武器池概率', weapon_gacha_system.stationary_rates())
