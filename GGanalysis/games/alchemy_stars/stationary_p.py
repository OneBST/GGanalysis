from GGanalysis.games.alchemy_stars import PITY_6STAR, P_5, P_4, P_3
from GGanalysis.markov import PriorityPityChain

# 调用预置工具
gacha_system = PriorityPityChain([PITY_6STAR, [0, P_5], [0, P_4], [0, P_3]])
print(gacha_system.stationary_rates())
