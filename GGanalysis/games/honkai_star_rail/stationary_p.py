from GGanalysis.games.honkai_star_rail import PITY_5STAR, PITY_4STAR, PITY_W5STAR, PITY_W4STAR
from GGanalysis.markov.markov_method_old import PriorityPitySystem

# 调用预置工具计算在五星四星耦合情况下的概率
common_gacha_system = PriorityPitySystem([PITY_5STAR, PITY_4STAR, [0, 1]])
print('常驻及角色池概率', common_gacha_system.get_stationary_p())
light_cone_gacha_system = PriorityPitySystem([PITY_W5STAR, PITY_W4STAR, [0, 1]])
print('光锥池概率',light_cone_gacha_system.get_stationary_p())