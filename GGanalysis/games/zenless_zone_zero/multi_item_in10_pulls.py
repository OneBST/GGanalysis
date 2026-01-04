import GGanalysis as gg
import GGanalysis.games.zenless_zone_zero as ZZZ
import numpy as np
ans_5 = gg.multi_item_rarity(ZZZ.PITY_5STAR, 10)
ans_4 = gg.multi_item_rarity(ZZZ.PITY_4STAR, 10)

print(ans_5)
print(ans_4)