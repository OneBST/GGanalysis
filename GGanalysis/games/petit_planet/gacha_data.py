"""星布谷地（Petit Planet）三测卡池数值。

一、服饰池（curio_clothing）
------------------------------
直接抽取服饰，按概率、兑换价、重复返还分成六组：彩、170 金、150 金、60 金、
紫、蓝，各组数量为 2、2、2、4、4、4。18 件首件兑换价合计 1420 星布。
彩、金共用十抽保底，命中任意彩金后计数归零；连续九抽未出则下一抽必中。
保底按各件基础概率的相对权重分配，可以重复，不保证彩色或新件。
首次获得解锁服饰，重复获得转为星布，星布用于兑换未拥有服饰。
同类池的保底计数和星布跨期保留；累计抽数返利按档位领取。

二、家具池（curio_furniture）
--------------------------------
直接抽取家具，彩、金、紫三档数量为 4、8、12，基础概率合计为 1.4%、5.6%、93%。
彩金共用十抽保底，命中后重置，保底按各件基础概率的相对权重分配。
首件兑换价为 70、30、6，重复抽取返还为 20、10、2；24 件首件合计 592 积木。
默认集齐要求其余 23 件成品和围栏创想卡。按有卡即可制作围栏处理，忽略制作材料；
围栏成品仍参与掉落和重复返还。零持有时直接补齐共需 626 积木。
首件解锁后，同款家具的重复兑换价降为该档重复抽取返还数。
同类池的保底计数和积木跨期保留，服饰与家具分别记录保底和兑换材料。

三、当季好礼普通服饰池（seasonal_clothing）
------------------------------------------
每次产出兑换材料，不直接抽取服饰。100 档概率为 2%，20、22、24 档合计为 8%，
三档均分；5、6、7、8 档各为 22.5%。最多十次获得一次 20、22 或 24 个材料。
每次材料产出均累加，不区分首次和重复；材料跨期保留。
服饰兑换分为 200、70、45 三个价格档，数量为 2、3、3，各一件合计 745。

四、常驻家具普通池（standard_furniture）
----------------------------------------
每次产出兑换材料，不直接抽取家具，产出数量档位及基础概率与普通服饰池相同。
最多十次获得一次 20、22 或 24 个材料，每次产出均累加。
默认收集 18 件家具成品，合计 852 小积木。另有三张各 70 小积木的创想卡，
计入配方时整张商店清单共 21 项，合计 1062。
本池与家具池使用不同兑换材料，分别计算余额和兑换支出。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ItemGroup:
    """同档且同概率、同兑换价、同重复返还的一组物品。

    count 为物品数，probability 为单件概率，exchange_cost 为单件首件兑换价。
    duplicate_refund 为重复抽取返还，repeat_exchange_cost 为解锁后的同款兑换价；
    None 表示该组不配置重复兑换价格。
    """

    rarity: str
    count: int
    probability: float
    exchange_cost: int
    duplicate_refund: int
    repeat_exchange_cost: int | None = None


# ---- 两个物品池共用的保底参数 ----
# 彩金共享保底，获得任意彩金后重置；两个池的计数分别继承。
CURIO_HIGH_PROBABILITY = 0.07
CURIO_HARD_PITY = 10
CURIO_PITY_RARITIES = ("彩", "金")
# 索引 0 占位，索引 k 为上次彩金之后第 k 抽的彩金概率。
CURIO_PITY_PROBABILITIES = (0.0,) + (CURIO_HIGH_PROBABILITY,) * 9 + (1.0,)


# ---- 卡池一：服饰池 ----
# 组键只表达档次和数值差异，同组内物品可按编号区分。
CLOTHING_GROUPS = {
    "rainbow": ItemGroup("彩", 2, 0.00035, 220, 20),
    "gold_170": ItemGroup("金", 2, 0.006825, 170, 15),
    "gold_150": ItemGroup("金", 2, 0.006825, 150, 15),
    "gold_60": ItemGroup("金", 4, 0.0105, 60, 10),
    "purple": ItemGroup("紫", 4, 0.081375, 20, 4),
    "blue": ItemGroup("蓝", 4, 0.151125, 5, 1),
}
# 默认目标为 CLOTHING_GROUPS 的全部首件。魔发泡泡是额外兑换的挑染耗材。
CLOTHING_EXCHANGE_ONLY_TIERS = {15: 1}
# (累计抽数门槛, 本档露芽奖励)，奖励逐档累加。
CLOTHING_PULL_REWARDS = (
    (5, 100), (25, 200), (60, 200), (95, 200), (130, 200), (160, 200),
)
# 综合概率的四位小数百分数转换为 0～1 小数。
CLOTHING_OVERALL_PROBABILITIES = {"彩金": 0.135654, "紫蓝": 0.864346}
# 各组内单件的保底条件概率；乘该组 count 后为整组保底概率。
CLOTHING_GUARANTEE_PROBABILITIES = {
    key: group.probability / CURIO_HIGH_PROBABILITY
    for key, group in CLOTHING_GROUPS.items() if group.rarity in CURIO_PITY_RARITIES
}


# ---- 卡池二：家具池 ----
# 解锁首件后，再次兑换同款的价格等于重复抽取返还。
FURNITURE_GROUPS = {
    "rainbow": ItemGroup("彩", 4, 0.0035, 70, 20, 20),
    "gold": ItemGroup("金", 8, 0.007, 30, 10, 10),
    "purple": ItemGroup("紫", 12, 0.0775, 6, 2, 2),
}
# 紫档中有一件围栏可由创想卡制作，仅其余紫档成品需要直接收集。
FURNITURE_DEFAULT_TARGET_COUNTS = {key: group.count - int(key == "purple")
                                   for key, group in FURNITURE_GROUPS.items()}
FURNITURE_FENCE_CARD_COST = 40
FURNITURE_EXCHANGE_ONLY_TIERS = {FURNITURE_FENCE_CARD_COST: 1}
FURNITURE_OVERALL_PROBABILITIES = {"彩": 0.027131, "金": 0.108523, "紫": 0.864346}
# 各组内单件的保底条件概率。
FURNITURE_GUARANTEE_PROBABILITIES = {
    key: group.probability / CURIO_HIGH_PROBABILITY
    for key, group in FURNITURE_GROUPS.items() if group.rarity in CURIO_PITY_RARITIES
}


# ---- 两个普通材料池共用的产出档位 ----
# (材料数量, 基础概率)。20、22、24 档共 8%，单档显示百分数为 2.6667%。
# 计算使用 8% 三等分，不对显示值之和强制归一化。
MATERIAL_REWARDS = (
    (100, 0.02), (24, 0.08 / 3), (22, 0.08 / 3), (20, 0.08 / 3),
    (8, 0.225), (7, 0.225), (6, 0.225), (5, 0.225),
)
MATERIAL_DISPLAY_PERCENT = {
    100: 2.0000, 24: 2.6667, 22: 2.6667, 20: 2.6667,
    8: 22.5000, 7: 22.5000, 6: 22.5000, 5: 22.5000,
}
# 最多十次产出一次 20、22 或 24 个材料；100 档单独列出。
MATERIAL_HARD_PITY = 10
MATERIAL_PITY_AMOUNTS = (20, 22, 24)


# ---- 卡池三：当季好礼普通服饰池 ----
# 单件兑换价 -> 该价位物品数；合计 8 件，全部兑换需要 745 个材料。
SEASONAL_CLOTHING_EXCHANGE_TIERS = {200: 2, 70: 3, 45: 3}
SEASONAL_CLOTHING_OVERALL_PROBABILITIES = {
    "100": 0.018664, "20/22/24": 0.141440, "5/6/7/8": 0.839896,
}


# ---- 卡池四：常驻家具普通池 ----
# 单件兑换价 -> 该价位物品数；合计 21 项，全部兑换需要 1062 个材料。
STANDARD_FURNITURE_ITEM_TIERS = {116: 2, 50: 7, 30: 9}
STANDARD_FURNITURE_RECIPE_TIERS = {70: 3}
STANDARD_FURNITURE_EXCHANGE_TIERS = STANDARD_FURNITURE_ITEM_TIERS | STANDARD_FURNITURE_RECIPE_TIERS


# 单期从零持有、零余额、零保底开始，够换齐默认目标即停。
# 上新池的最坏序列为九次同一低档、一件同一最低价金档，循环至补齐。
# 服饰第 719、720 抽的累计兑换进度为 1412、1422。
# 家具最低档反复出围栏，首次无兑换进度；第 219、220 抽的进度为 624、634。
# 普通池最坏每十抽产出 9 × 5 + 20，余下每抽产出 5。
SINGLE_SEASON_GUARANTEED_PULLS: dict[str, int] = {
    "curio_clothing": 720,
    "curio_furniture": 220,
    "seasonal_clothing": 116,
    "standard_furniture": 132,
}


__all__ = [
    "ItemGroup", "CURIO_HIGH_PROBABILITY", "CURIO_HARD_PITY", "CURIO_PITY_RARITIES",
    "CURIO_PITY_PROBABILITIES", "CLOTHING_GROUPS", "CLOTHING_EXCHANGE_ONLY_TIERS",
    "CLOTHING_PULL_REWARDS", "CLOTHING_OVERALL_PROBABILITIES",
    "CLOTHING_GUARANTEE_PROBABILITIES", "FURNITURE_GROUPS",
    "FURNITURE_EXCHANGE_ONLY_TIERS", "FURNITURE_OVERALL_PROBABILITIES",
    "FURNITURE_DEFAULT_TARGET_COUNTS", "FURNITURE_FENCE_CARD_COST",
    "FURNITURE_GUARANTEE_PROBABILITIES", "MATERIAL_REWARDS", "MATERIAL_DISPLAY_PERCENT",
    "MATERIAL_HARD_PITY", "MATERIAL_PITY_AMOUNTS", "SEASONAL_CLOTHING_EXCHANGE_TIERS",
    "SEASONAL_CLOTHING_OVERALL_PROBABILITIES", "STANDARD_FURNITURE_EXCHANGE_TIERS",
    "STANDARD_FURNITURE_ITEM_TIERS", "STANDARD_FURNITURE_RECIPE_TIERS",
    "SINGLE_SEASON_GUARANTEED_PULLS",
]
