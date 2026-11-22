"""三测上新池的物品分组、保底转移和兑换首次命中分布。"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from GGanalysis import FiniteDist, HitProcess, HitTransitionBuilder, StateSpace
from . import gacha_data as data

Item = tuple[str, int]
POOLS = {"curio_clothing": data.CLOTHING_GROUPS, "curio_furniture": data.FURNITURE_GROUPS}
HIGH_RATE = np.array(data.CURIO_PITY_PROBABILITIES[1:])


@dataclass(frozen=True)
class Group:
    count: int
    owned: int
    high: bool
    probability: float
    price: int  # 非目标首件的有效收益为零
    refund: int
    items: tuple[Item, ...]


def collection_groups(pool: str, target: set[Item] | None, owned: set[Item],
                      exchange: bool, fence_card: bool) -> tuple[list[Group], int]:
    """组内从 1 编号，最后一件紫家具为围栏；按本次计算实际需要的属性合并。"""
    items = defaultdict(list)
    acquired = Counter()
    for name, item in POOLS[pool].items():
        for index in range(1, item.count + 1):
            key = (name, index)
            wanted = target is None or key in target
            if fence_card and name == "purple" and index > data.FURNITURE_DEFAULT_TARGET_COUNTS["purple"]:
                wanted = False
            if not exchange and not wanted:
                continue
            price = item.exchange_cost * wanted if exchange else 1
            group = (item.rarity in data.CURIO_PITY_RARITIES, item.probability,
                     price, item.duplicate_refund if exchange else 0)
            items[group].append(key)
            acquired[group] += key in owned
    groups = [Group(len(members), acquired[key], *key, tuple(members))
              for key, members in items.items()]
    return groups, sum((g.count - g.owned) * g.price for g in groups)


def first_hit_process(groups: list[Group]) -> HitProcess:
    """纯抽组计数与保底的稀疏过程，命中时保留结束保底。"""
    shape = tuple(g.count - g.owned + 1 for g in groups) + (len(HIGH_RATE),)
    space = StateSpace.from_shape(shape)
    boundary = StateSpace.from_shape([len(HIGH_RATE)])
    builder = HitTransitionBuilder(space, boundary, np.arange(boundary.N))
    grid = np.indices(shape).reshape(len(shape), -1)
    ids = np.arange(space.N)
    h = grid[-1]
    missing = sum(shape[:-1]) - len(groups) - grid[:-1].sum(axis=0)
    for high, rate in ((True, HIGH_RATE), (False, 1 - HIGH_RATE)):
        next_h = np.zeros_like(h) if high else np.minimum(h + 1, boundary.N - 1)
        repeat = rate[h].copy()
        for j, group in enumerate(groups):
            if group.high != high:
                continue
            p = (shape[j] - 1 - grid[j]) * group.probability * rate[h] / rate[0]
            repeat -= p
            mask = (missing > 1) & (p > 0)
            dest = ids + int(np.prod(shape[j+1:])) + next_h - h
            builder.add_miss_many(ids[mask], dest[mask], p[mask])
            mask = (missing == 1) & (p > 0)
            builder.add_hit_many(ids[mask], next_h[mask], p[mask])
        mask = (missing > 0) & (repeat > 0)
        builder.add_miss_many(ids[mask], (ids + next_h - h)[mask], repeat[mask])
        # 全集齐格不从未完成格进入，只用于满足过程的列守恒。
        mask = missing == 0
        builder.add_hit_many(ids[mask], next_h[mask], rate[h[mask]])
    return builder.build()


def pure_stats(process: HitProcess, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """未截断的平均抽数及各结束保底概率；利用逐件收集时的三角块顺序。"""
    miss = process.miss
    visits = spsolve((sparse.eye(miss.shape[0], format="csc") - miss).tocsc(),
                     starts, permc_spec="NATURAL")
    return visits.sum(axis=0), process.hit @ visits


def tier_progress(groups: list[Group], threshold: int, limit: int, unit: int,
                  tier_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """给定本档已出 k 次，计算有效收益及下一次跨过各门槛的概率。"""
    shape = tuple(g.count - g.owned + 1 for g in groups)
    coords = np.indices(shape).reshape(len(groups), -1)
    size = int(np.prod(shape))
    branches = []
    for j, g in enumerate(groups):
        stride = int(np.prod(shape[j+1:]))
        owned = coords[j] + g.owned
        for count, offset, gain in ((owned, 0, g.refund),
                                    (g.count - owned, stride, g.price)):
            src = np.flatnonzero(count)
            branches.append((src, src + offset, count[src, None] * (g.probability / tier_rate),
                             gain // unit))
    mass = np.zeros((size, threshold))
    mass[0, 0] = 1
    totals = np.zeros((limit + 1, threshold))
    crossing = np.zeros_like(totals)
    for k in range(limit + 1):
        totals[k] = mass.sum(axis=0)
        if k == limit or not mass.any():
            break
        following = np.zeros_like(mass)
        for src, dest, p, gain in branches:
            flow = mass[src] * p
            if gain:
                # 收益原为 x 时，本次跨过 x+1 到 x+gain，逐门槛直接累计概率。
                crossing[k] += np.convolve(flow.sum(axis=0), np.ones(gain))[:threshold]
            if gain < threshold:
                following[dest, gain:] += flow[:, :threshold-gain]
        mass = following
    return totals, crossing


def exchange_dist(groups: list[Group], deficit: int, pity: int, limit: int) -> FiniteDist:
    """条件于彩金次数，两档的物品序列独立；按各档持有组合分别递推。"""
    unit = reduce(gcd, (value for g in groups for value in (g.price, g.refund)))
    threshold = (deficit + unit - 1) // unit
    high, high_cross = tier_progress([g for g in groups if g.high], threshold, limit,
                                     unit, HIGH_RATE[0])
    low, low_cross = tier_progress([g for g in groups if not g.high], threshold, limit,
                                   unit, 1 - HIGH_RATE[0])
    # counts[i,h] 为已经出了 i 次彩金且保底为 h 的概率。
    counts = np.zeros((limit + 1, len(HIGH_RATE)))
    counts[0, pity] = 1
    pmf = np.zeros(limit + 1)
    for n in range(limit):
        i = np.arange(n + 1)
        j = n - i
        high_prob = (counts[:n+1] * HIGH_RATE).sum(axis=1)
        low_by_pity = counts[:n+1, :-1] * (1 - HIGH_RATE[:-1])
        # 对侧贡献 x 时，本侧只须首次跨过 threshold-x。
        hit_high = np.einsum("ij,ij->i", low[j], high_cross[:n+1, ::-1])
        hit_low = np.einsum("ij,ij->i", high[:n+1], low_cross[j, ::-1])
        pmf[n+1] = high_prob @ hit_high + low_by_pity.sum(axis=1) @ hit_low
        counts[:n+1, 1:] = low_by_pity
        counts[0, 0] = 0
        counts[1:n+2, 0] = high_prob
    # 截止后直接汇总两档收益之和仍低于门槛的质量，避免用 1-CDF 消减尾数。
    below = np.cumsum(low, axis=1)
    survival = np.einsum("ij,ij->i", high, below[::-1, ::-1])
    tail = float(counts.sum(axis=1) @ survival)
    return FiniteDist(pmf, trim_tail_zeros=False, tail_mass=tail)


def minimum_progress(groups: list[Group], limit: int) -> np.ndarray:
    """本档抽 k 次的最低有效收益。枚举重复掉落的那一件，其余便宜首件各取一次。"""
    draws = np.arange(1, limit + 1)
    minimum = np.full(limit, np.iinfo(np.int64).max, dtype=np.int64)
    for j, chosen in enumerate(groups):
        for is_new, count in ((True, chosen.count-chosen.owned), (False, chosen.owned)):
            if count == 0:
                continue
            first = chosen.price if is_new else chosen.refund
            savings = []
            for i, group in enumerate(groups):
                available = group.count - group.owned - int(is_new and i == j)
                if group.price < chosen.refund:
                    savings.extend([chosen.refund-group.price] * available)
            saved = np.cumsum([0] + sorted(savings, reverse=True))
            progress = first + (draws-1)*chosen.refund - saved[np.minimum(draws-1, len(savings))]
            minimum = np.minimum(minimum, progress)
    return np.r_[0, minimum]


MATERIAL_TIERS = {"seasonal_clothing": data.SEASONAL_CLOTHING_EXCHANGE_TIERS,
                  "standard_furniture": data.STANDARD_FURNITURE_ITEM_TIERS}
Exchange = Literal["optimal", "never"]


def material_price(pool: str, cost: int | None) -> int:
    return cost if cost is not None else sum(c*n for c, n in MATERIAL_TIERS[pool].items())


def curio_goal(pool: str, target: set[Item] | None, owned: set[Item] | None, redeem: bool,
               fence_card: bool | None, has_card: bool) -> tuple[list[Group], int | float]:
    if fence_card is None:
        fence_card = pool == "curio_furniture" and target is None and redeem
    groups, remaining = collection_groups(pool, target, owned or set(), redeem, fence_card)
    if fence_card and not has_card:
        remaining += data.FURNITURE_FENCE_CARD_COST if redeem else np.inf
    return groups, remaining


def material_dist(deficit: int, initial: np.ndarray, limit: int) -> FiniteDist:
    """从余额与保底的联合分布出发，求普通池首次够兑换的抽数。"""
    hard = data.MATERIAL_HARD_PITY
    space = StateSpace.from_shape([deficit, hard])
    builder = HitTransitionBuilder(space, StateSpace.from_shape([1]), [0])
    ids = np.arange(space.N)
    amount, h = np.divmod(ids, hard)
    for gain, base in data.MATERIAL_REWARDS:
        resets = gain in data.MATERIAL_PITY_AMOUNTS
        p = np.where(h == hard-1, 1 / len(data.MATERIAL_PITY_AMOUNTS) if resets else 0, base)
        next_h = 0 if resets else h + 1
        dest = (amount + gain) * hard + next_h
        hit = amount + gain >= deficit
        mask = ~hit & (p > 0)
        builder.add_miss_many(ids[mask], dest[mask], p[mask])
        mask = hit & (p > 0)
        builder.add_hit_many(ids[mask], np.zeros(mask.sum(), dtype=int), p[mask])
    start = np.zeros((deficit, hard))
    rows = min(deficit, len(initial))
    start[:rows] = initial[:rows]
    active = start.sum()
    zero = initial[deficit:].sum()
    if not active:
        return FiniteDist([1.], exp=0, tail_mass=0.)
    dist, tail = builder.build().first_hit(start.ravel()/active, limit, return_remaining=True)
    pmf = dist.coeff[:, 0]*active
    pmf[0] = zero
    return FiniteDist(pmf, tail_mass=tail*active)

