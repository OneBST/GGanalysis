"""活动盲盒的加权不放回抽取与递增花费。"""

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from GGanalysis.basic_models import GachaModel
from GGanalysis.distribution_1d import FiniteDist

__all__ = ["NTEBlindBoxModel"]


class NTEBlindBoxModel(GachaModel):
    """按剩余奖品数量乘权重抽取，命中后移除一份奖品。

    Parameters
    ----------
    weights : sequence
        每类奖品的非负权重；一维表示全程固定，二维的第 k 行表示第 k+1
        抽的权重，可用零权重表达某些抽次不能获得该奖品。不是逐抽命中概率。
    counts : sequence of int
        各类奖品初始份数，每份可以是一个道具包。同权重、同规则的非目标
        奖品可合并为一类；需要分别查询的奖品应分开定义。
    costs : mapping of str to sequence of int, optional
        花费单位名到逐抽价格的映射，序列从第一抽开始，不含索引零占位。
        建议方斯采用「万方斯」单位，避免创建千万长度的稀疏一维数组。

    Notes
    -----
    精确枚举剩余数量状态，不假设各抽独立，不用边际概率代替状态条件概率。
    命中目标后停止，不自动发放剩余奖品、不扣除奖励价值；参数应按期次配置。
    """

    def __init__(self, weights: Sequence, counts: Sequence[int],
                 costs: Mapping[str, Sequence[int]] | None = None) -> None:
        quantities = np.asarray(counts)
        if (quantities.ndim != 1 or not quantities.size
                or not np.issubdtype(quantities.dtype, np.integer)
                or np.any(quantities <= 0)):
            raise ValueError("counts 必须是一维正整数序列")
        self.counts = tuple(int(n) for n in quantities)
        self.total = sum(self.counts)
        values = np.asarray(weights, dtype=float)
        if values.shape == (len(self.counts),):
            values = np.broadcast_to(values, (self.total, len(self.counts)))
        if (values.shape != (self.total, len(self.counts))
                or not np.all(np.isfinite(values)) or np.any(values < 0)):
            raise ValueError("weights 必须是每类权重或逐抽权重矩阵，且有限非负")
        self.weights = values.copy()
        self.weights.flags.writeable = False
        self.costs = {}
        for unit, prices in (costs or {}).items():
            prices = np.asarray(prices)
            if (unit == "pull" or prices.shape != (self.total,)
                    or not np.issubdtype(prices.dtype, np.integer) or np.any(prices < 0)):
                raise ValueError("价格必须是覆盖全部抽次的非负整数序列，单位名不能为 pull")
            self.costs[unit] = tuple(int(price) for price in prices)

    def __call__(self, target: int, item_num: int = 1, unit: str = "pull",
                 remaining: Sequence[int] | None = None) -> FiniteDist:
        """返回从指定剩余奖池再获得目标奖品的花费分布。

        Parameters
        ----------
        target : int
            目标奖品类别的零基索引。
        item_num : int, optional
            从当前状态再获得的目标份数，默认 1；零份返回零花费分布。
        unit : str, optional
            ``pull`` 表示活动抽取次数；其他名称必须在构造时的 costs 中定义。
        remaining : sequence of int, optional
            当前每类奖品的剩余份数，默认完整奖池。已抽次数由总份数差得到，
            因而继续使用原进度的权重与价格，而不是从第一抽重新计价。

        Returns
        -------
        FiniteDist
            索引是从当前状态起新增的花费；exp 是该单位的期望。获取奖品的
            历史需用 remaining 指明，单独知道已抽次数一般不足以决定概率。
        """
        if not isinstance(target, (int, np.integer)) or not 0 <= target < len(self.counts):
            raise ValueError("target 必须是有效的奖品类别索引")
        initial = np.asarray(self.counts if remaining is None else remaining)
        if (initial.shape != (len(self.counts),)
                or not np.issubdtype(initial.dtype, np.integer)
                or np.any(initial < 0) or np.any(initial > self.counts)):
            raise ValueError("remaining 必须逐项位于零与初始份数之间")
        if (not isinstance(item_num, (int, np.integer))
                or not 0 <= item_num <= initial[target]):
            raise ValueError("item_num 必须为不超过剩余目标份数的非负整数")
        if unit != "pull" and unit not in self.costs:
            raise ValueError(f"未定义花费单位：{unit}")
        if item_num == 0:
            return FiniteDist.delta(0)

        initial = tuple(int(n) for n in initial)
        remaining_total = sum(initial)
        drawn = self.total - remaining_total
        threshold = initial[target] - item_num
        pmf = np.zeros(remaining_total + 1)
        states = {initial: 1.0}
        for step in range(1, remaining_total + 1):
            next_states = defaultdict(float)
            weights = self.weights[drawn + step - 1]
            for state, mass in states.items():
                weighted_counts = np.asarray(state) * weights
                denominator = weighted_counts.sum()
                if denominator <= 0:
                    raise ValueError("可达奖池的总权重为零，请检查逐抽权重配置")
                for index in np.flatnonzero(weighted_counts):
                    probability = mass * weighted_counts[index] / denominator
                    updated = list(state)
                    updated[index] -= 1
                    if updated[target] == threshold:
                        pmf[step] += probability
                    else:
                        next_states[tuple(updated)] += probability
            states = next_states
            if not states:
                break
        if not np.isclose(pmf.sum(), 1.0, rtol=0, atol=1e-12):
            raise ArithmeticError("盲盒递推概率质量不守恒")
        if unit == "pull":
            return FiniteDist(pmf, tail_mass=0.0)
        prices = self.costs[unit][drawn:]
        cumulative_costs = np.r_[0, np.cumsum(prices)]
        return FiniteDist(np.bincount(cumulative_costs, weights=pmf), tail_mass=0.0)

    def expectations(self, target: int, item_num: int = 1,
                     remaining: Sequence[int] | None = None) -> dict[str, float]:
        """返回活动抽数及各支付单位的期望，不构造大金额索引数组。

        参数与 ``__call__`` 的同名参数一致；返回键 ``pull`` 和 costs 中的
        各单位名。各货币结果分别假设全程使用该货币，不相加、不扣奖励。
        """
        dist = self(target, item_num, remaining=remaining)
        drawn = 0 if remaining is None else self.total - sum(remaining)
        result = {"pull": dist.exp}
        for unit, prices in self.costs.items():
            cumulative = np.r_[0, np.cumsum(prices[drawn:])]
            result[unit] = float(np.dot(dist.dist, cumulative[:len(dist)]))
        return result
