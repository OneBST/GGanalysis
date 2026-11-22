"""同模板无限换季的平均抽数及资源约束下的平稳停手策略。"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import spsolve

from GGanalysis import FiniteDist, MarkovTransition, StateSpace, stationary_solve
from . import gacha_data as data
from .collection import (HIGH_RATE, MATERIAL_TIERS, Exchange, Group, Item, curio_goal,
                         first_hit_process, material_dist, material_price, pure_stats)

__all__ = ["LongRunResult", "long_run"]


@dataclass
class LongRunResult:
    """每季平均抽数；兑换策略在有平稳访问量的状态上给出继续抽的概率。

    group_items 依次列出动作表各轴的物品。draw_probability 以各组拥有数、
    当前保底为下标；NaN 表示该状态在所得平稳解中没有访问量。
    """
    mean_pulls: float
    group_items: tuple[tuple[Item, ...], ...] = ()
    draw_probability: np.ndarray | None = None
    stop_dist: FiniteDist | None = None


def long_run(pool: str, exchange: Exchange = "optimal", *, target: set[Item] | None = None,
             fence_card: bool | None = None, material_cost: int | None = None,
             max_pulls: int | None = None) -> LongRunResult:
    """计算无限次重复同模板、每期全新且继承保底时的长期每季平均。

    Parameters
    ----------
    pool, exchange, target, fence_card, material_cost
        同 single_season。普通池按每季重新提供同一价格清单处理。
        上新池兑换使用长期平均收支约束，忽略返利、单季预算和库存上限。
    max_pulls : int, optional
        若给定，同时计算平稳策略每期恰在各抽数停手的分布，截止后返回尾概率。

    Returns
    -------
    LongRunResult
        纯抽返回未截断的长期均值。上新兑换返回资源松弛最优均值及平稳策略；
        实际余额不足以补齐时仍须继续抽，有限余额执行的成本可能更高。
    """
    redeem = exchange == "optimal"
    def trivial(mean: float) -> LongRunResult:
        stop = None
        if max_pulls is not None:
            reached = float(mean == 0)
            stop = FiniteDist([reached], exp=mean, tail_mass=1-reached)
        return LongRunResult(mean, stop_dist=stop)

    if pool in MATERIAL_TIERS:
        cost = material_price(pool, material_cost)
        if not cost or not redeem:
            return trivial(np.inf if cost else 0.)
        rewards = data.MATERIAL_REWARDS
        hits = data.MATERIAL_PITY_AMOUNTS
        base = sum(p for gain, p in rewards if gain in hits)
        hard = data.MATERIAL_HARD_PITY
        pity_mass = (1-base)**np.arange(hard)
        pity_mass /= pity_mass.sum()
        # 每次发放 g 份材料跨过 g 个单位门槛，对应结余 0,...,g-1。
        # 以门槛为观察点计数，得到季初结余与继承保底的联合平稳分布。
        ends = np.zeros((max(gain for gain, _ in rewards), hard))
        for gain, p in rewards:
            if gain in hits:
                ends[:gain, 0] += p*pity_mass[:-1].sum() + pity_mass[-1]/len(hits)
            else:
                ends[:gain, 1:] += p*pity_mass[:-1]
        mean_gain = ends.sum()
        stop = material_dist(cost, ends/mean_gain, max_pulls) if max_pulls is not None else None
        return LongRunResult(cost / mean_gain, stop_dist=stop)

    groups, cost = curio_goal(pool, target, set(), redeem, fence_card, False)
    if cost == 0 or cost == np.inf:
        return trivial(cost)
    hard = len(HIGH_RATE)
    if not redeem:
        process = first_hit_process(groups)
        start = np.zeros((process.full_space.N, hard))
        start[:hard] = np.eye(hard)
        means, kernel = pure_stats(process, start)
        mu = stationary_solve(MarkovTransition(StateSpace.from_shape([hard]), kernel))
        mean = float(means @ mu)
        stop = None
        if max_pulls is not None:
            dist, tail = process.first_hit(start @ mu, max_pulls, return_remaining=True)
            stop = dist.marginal_cost(tail_mass=tail)
            stop.exp = mean
        return LongRunResult(mean, stop_dist=stop)

    # 占用量 LP：draw[s]、stop[s] 为每季在 s 抽取和结束的平均次数。
    # 流入等于流出，每季恰好结束一次，平均有效收益至少覆盖本季首件总价。
    shape = tuple(g.count+1 for g in groups) + (hard,)
    grid = np.indices(shape).reshape(len(shape), -1)
    h = grid[-1]
    size = len(h)
    ids = np.arange(size)
    reward = np.zeros(size)
    rows, cols, weights = [], [], []
    for j, g in enumerate(groups):
        rate = HIGH_RATE if g.high else 1-HIGH_RATE
        p = g.probability * rate[h] / rate[0]
        next_h = np.zeros_like(h) if g.high else np.minimum(h+1, hard-1)
        stride = int(np.prod(shape[j+1:]))
        for count, offset, gain in ((grid[j], 0, g.refund),
                                    (g.count-grid[j], stride, g.price)):
            prob = count*p
            keep = prob > 0
            cols.append(ids[keep])
            rows.append((ids + next_h - h + offset)[keep])
            weights.append(prob[keep])
            reward += prob*gain
    P = sparse.coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(size, size)).tocsc()
    return _season_mixture(P, reward, groups, shape, cost, max_pulls)


def _season_mixture(P: sparse.csc_matrix, reward: np.ndarray, groups: list[Group],
                    shape: tuple[int, ...], cost: int, limit: int | None) -> LongRunResult:
    """列生成：主问题只平衡十种季初保底和资金，定价子问题为一期最优停抽。"""
    hard = shape[-1]
    size = len(reward)
    h = np.arange(size) % hard
    start = np.eye(size, hard)
    identity = sparse.eye(size, format="csc")
    draw_p = P.T.tocsr()
    # 以整期停止策略为列；另加全重复时持续抽取的极限收益。
    pity_mass = np.r_[1., np.cumprod(1-HIGH_RATE[:-1])]
    pity_mass /= pity_mass.sum()
    farm_gain = pity_mass @ reward[-hard:]
    columns = [np.r_[np.zeros(hard), 0.]]
    gains, costs = [farm_gain], [1.]
    policies = []

    def add_policy(draw):
        system = (identity - P @ sparse.diags(draw.astype(float))).tocsc()
        visits = spsolve(system, start, permc_spec="NATURAL")
        pulls = visits * draw[:, None]
        ends = (visits - pulls).reshape(-1, hard, hard).sum(axis=0)
        columns.extend(np.column_stack([(np.eye(hard)-ends).T, np.ones(hard)]))
        costs.extend(pulls.sum(axis=0))
        gains.extend(reward @ pulls)
        policies.append(draw.copy())

    add_policy(np.zeros(size, dtype=bool))
    add_policy(np.arange(size) < hard)
    while True:
        lp = linprog(costs, A_eq=np.array(columns).T, b_eq=np.r_[np.zeros(hard),1.],
                     A_ub=-np.array(gains)[None,:], b_ub=[-cost], method="highs",
                     options={"dual_feasibility_tolerance": 1e-10,
                              "primal_feasibility_tolerance": 1e-10})
        if not lp.success:
            raise RuntimeError(lp.message)
        # 定价：最大化 -抽数 + 币价×收益 + 期末保底价值。
        price = -lp.ineqlin.marginals[0]
        boundary = -lp.eqlin.marginals[:hard]
        beta = -lp.eqlin.marginals[-1]
        draw = np.zeros(size, dtype=bool)
        immediate = price*reward-1
        while True:
            system = identity - sparse.diags(draw.astype(float)) @ draw_p
            value = spsolve(system.tocsc(), np.where(draw, immediate, boundary[h]),
                            permc_spec="NATURAL")
            advantage = immediate + draw_p @ value - boundary[h]
            improved = np.where(abs(advantage) <= 1e-10, draw, advantage > 0)
            if np.array_equal(draw, improved):
                break
            draw = improved
        if np.max(value[:hard]-boundary) <= beta + 1e-8:
            break
        add_policy(draw)

    # 最优列按季初保底加权，恢复每个状态的访问量和随机停抽概率。
    total = np.zeros(size)
    pulls = np.zeros(size)
    for draw, weights in zip(policies, lp.x[1:].reshape(-1, hard)):
        if not weights.any():
            continue
        system = identity - P @ sparse.diags(draw.astype(float))
        visits = spsolve(system.tocsc(), start @ weights, permc_spec="NATURAL")
        total += visits
        pulls += visits*draw
    total[-hard:] += lp.x[0]*pity_mass
    pulls[-hard:] += lp.x[0]*pity_mass
    policy = np.divide(pulls, total, out=np.full(size, np.nan), where=total > 0)
    stop = None
    if limit is not None and lp.x[0] == 0:
        prob = np.nan_to_num(policy)
        alive = start @ lp.x[1:].reshape(-1, hard).sum(axis=0)
        pmf = np.zeros(limit+1)
        for n in range(limit+1):
            pmf[n] = alive @ (1-prob)
            alive *= prob
            if n < limit:
                alive = P @ alive
        stop = FiniteDist(pmf, exp=lp.fun, tail_mass=alive.sum())
    return LongRunResult(lp.fun, tuple(g.items for g in groups), policy.reshape(shape), stop)

