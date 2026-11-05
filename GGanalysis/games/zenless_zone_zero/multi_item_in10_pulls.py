from GGanalysis.markov import stationary_item_count_distribution
import GGanalysis.games.zenless_zone_zero as ZZZ
# 起始保底进度按长期平稳分布混合，结果不是指定当前垫抽状态的条件分布。
ans_5 = stationary_item_count_distribution(ZZZ.PITY_5STAR, 10)
ans_4 = stationary_item_count_distribution(ZZZ.PITY_4STAR, 10)

print(ans_5)
print(ans_4)
