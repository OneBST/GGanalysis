from GGanalysis.games.zenless_zone_zero import PITY_5STAR, PITY_4STAR, PITY_W5STAR, PITY_W4STAR
from GGanalysis.stationary_distribution_method import PriorityPitySystem

# 调用预置工具计算在五星四星耦合情况下的概率
common_gacha_system = PriorityPitySystem([PITY_5STAR, PITY_4STAR, [0, 1]], remove_pity=True)
print('常驻及角色池概率', common_gacha_system.get_stationary_p())
print('常驻及角色池分布', common_gacha_system.get_type_distribution(1))

weapon_gacha_system = PriorityPitySystem([PITY_W5STAR, PITY_W4STAR, [0, 1]], remove_pity=True)
print('音擎及邦布池概率', weapon_gacha_system.get_stationary_p())
print('音擎及邦布池分布', weapon_gacha_system.get_type_distribution(1))