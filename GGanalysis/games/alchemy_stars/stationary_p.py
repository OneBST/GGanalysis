from GGanalysis.games.alchemy_stars import PITY_6STAR, P_5, P_4, P_3
from GGanalysis.stationary_distribution_method import PriorityPitySystem

# 调用预置工具
gacha_system = PriorityPitySystem([PITY_6STAR, [0, P_5], [0, P_4, P_4, P_4, P_4, P_4, P_4, P_4, P_4, P_4, 1], [0, P_3]])
print(gacha_system.get_stationary_p())