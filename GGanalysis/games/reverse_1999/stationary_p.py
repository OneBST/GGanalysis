from GGanalysis.games.reverse_1999 import PITY_6STAR, P_5, P_4, P_3, P_2
from GGanalysis.markov import PriorityPityChain

# 调用预置工具
gacha_system = PriorityPityChain([PITY_6STAR, [0, P_5], [0, P_4, P_4, P_4, P_4, P_4, P_4, P_4, P_4, P_4, 1], [0, P_3], [0, P_2]])
print(gacha_system.stationary_rates())
