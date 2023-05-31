from GGanalysis.games.reverse_1999 import PITY_6STAR, P_5, P_4, P_3, P_2
from GGanalysis.stationary_distribution_method import PriorityPitySystem

# 调用预置工具，但实际上由于 1999 考虑有优先级的情况应该采用工具而不能直接按排除6星综合概率折算
gacha_system = PriorityPitySystem([PITY_6STAR, [0, P_5], [0, P_4], [0, P_3], [0, P_2]])
print(gacha_system.get_stationary_p())