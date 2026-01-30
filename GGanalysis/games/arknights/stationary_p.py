from GGanalysis.games.arknights import PITY_6STAR, PITY_5STAR
from GGanalysis.markov.markov_method_old import PriorityPitySystem

# 调用预置工具计算在六星五星耦合情况下的概率，六星会重置五星保底，remov_pity 应设置为 True
gacha_system = PriorityPitySystem([PITY_6STAR, PITY_5STAR], remove_pity=True)
print('六星及五星平稳概率', gacha_system.get_stationary_p())