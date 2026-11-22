"""星布谷地三测的单期集齐与无限同模板换季模型。"""

import numpy as np

from GGanalysis import FiniteDist
from . import gacha_data as data
from .collection import (HIGH_RATE, MATERIAL_TIERS, Exchange, Item, curio_goal, exchange_dist,
                         first_hit_process, material_dist, material_price, minimum_progress, pure_stats)
from .strategy import LongRunResult, long_run

__all__ = ["single_season", "guaranteed_pulls", "long_run", "LongRunResult"]



def single_season(pool: str, exchange: Exchange = "optimal", *, target: set[Item] | None = None,
                  owned: set[Item] | None = None, balance: int = 0, item_pity: int = 0,
                  fence_card: bool | None = None, has_card: bool = False,
                  material_cost: int | None = None, max_pulls: int | None = None) -> FiniteDist:
    """给定起点，返回首次自然抽齐或有能力一次补齐目标的抽数分布。

    Parameters
    ----------
    pool : str
        卡池键，见游戏文档。
    exchange : {"optimal", "never"}
        够兑换即停，或全程禁止兑换。后者默认只看可抽成品。
    target, owned : set of (str, int), optional
        上新池的目标、已解锁物品。组键及组内从 1 开始的编号唯一标识一件。
        target 默认全套，owned 默认空。家具围栏为 ("purple", 12)。
    balance, item_pity : int
        当前兑换币与连续未命中保底目标的抽数。
    fence_card, has_card : bool
        家具是否以配方覆盖围栏，以及是否已有卡。默认仅在家具全套兑换时要求配方。
    material_cost : int, optional
        普通池尚需购买的成品总价（扣除余额前），默认本期成品清单总价。
    max_pulls : int, optional
        计算截止，默认各池空仓全收集的兑换保证抽数。

    Returns
    -------
    FiniteDist
        下标为完成抽数，tail_mass 为截止后未完成概率（含不可达）。
        上新池纯抽的 exp 另用未截断的首次命中方程求得。
    """
    limit = data.SINGLE_SEASON_GUARANTEED_PULLS[pool] if max_pulls is None else max_pulls
    redeem = exchange == "optimal"
    if pool in MATERIAL_TIERS:
        cost = material_price(pool, material_cost)
        if cost == 0 or (redeem and balance >= cost):
            return FiniteDist([1.], exp=0, tail_mass=0.)
        if not redeem:
            return FiniteDist([0.], exp=np.inf, tail_mass=1.)
        initial = np.eye(data.MATERIAL_HARD_PITY)[item_pity:item_pity+1]
        return material_dist(cost - balance, initial, limit)

    groups, remaining = curio_goal(pool, target, owned, redeem, fence_card, has_card)
    if remaining == np.inf:
        return FiniteDist([0.], exp=np.inf, tail_mass=1.)
    if redeem:
        remaining -= balance
    if remaining <= 0:
        return FiniteDist([1.], exp=0, tail_mass=0.)
    if redeem:
        return exchange_dist(groups, remaining, item_pity, limit)
    process = first_hit_process(groups)
    start = np.zeros(process.full_space.N)
    start[item_pity] = 1
    result, tail = process.first_hit(start, limit, return_remaining=True)
    dist = result.marginal_cost(tail_mass=tail)
    dist.exp = float(pure_stats(process, start)[0])
    return dist


def guaranteed_pulls(pool: str, *, target: set[Item] | None = None, owned: set[Item] | None = None,
                     balance: int = 0, item_pity: int = 0, fence_card: bool | None = None,
                     has_card: bool = False, material_cost: int | None = None) -> int:
    """计算单期允许兑换时保证完成的最小抽数；初态和目标参数同 single_season。"""
    if pool in MATERIAL_TIERS:
        deficit = material_price(pool, material_cost) - balance
        small = min(gain for gain, _ in data.MATERIAL_REWARDS)
        bonus = min(data.MATERIAL_PITY_AMOUNTS) - small
        def progress(n):
            return n*small + (n+item_pity)//data.MATERIAL_HARD_PITY*bonus
    else:
        groups, cost = curio_goal(pool, target, owned, True, fence_card, has_card)
        deficit = cost - balance
        limit = data.SINGLE_SEASON_GUARANTEED_PULLS[pool]
        high = minimum_progress([g for g in groups if g.high], limit)
        low = minimum_progress([g for g in groups if not g.high], limit)
        def progress(n):
            first = (n+item_pity)//len(HIGH_RATE)
            return np.min(high[first:n+1] + low[n-first::-1])
    lo, hi = 0, data.SINGLE_SEASON_GUARANTEED_PULLS[pool]
    while progress(hi) < deficit:
        hi *= 2
    while lo < hi:
        mid = (lo+hi)//2
        if progress(mid) >= deficit:
            hi = mid
        else:
            lo = mid+1
    return lo

