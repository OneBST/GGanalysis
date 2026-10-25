"""异环 Neverness to Everness 测试服棋盘抽卡模型。"""

from typing import Sequence, Union
from functools import cached_property

import numpy as np

from GGanalysis.basic_models import GachaModel, PityBernoulliModel
from GGanalysis.distribution_1d import FiniteDist
from GGanalysis.markov import MarkovTransition, StateSpace
from GGanalysis.state_distribution import ConvolutionMethod, StateKernel
from GGanalysis.games.neverness_to_everness.gacha_data import (
    ARC_DISC_S_PROBABILITY,
    ARC_DISC_UP_GIVEN_S,
    ARC_DISC_S_PITY,
    ARC_DISC_UP_PITY,
    CHARACTER_COSMETIC_RULES,
    CHARACTER_MAP_INFO,
    CHARACTER_MAP_PITY_INFO,
    CHARACTER_POOL_NORMAL_MAP_5,
    CHARACTER_POOL_PITY_MAP_5,
    PITY_START,
    STANDARD_MAP_INFO,
    STANDARD_MAP_PITY_INFO,
    STANDARD_POOL_NORMAL_MAP_5,
    STANDARD_POOL_PITY_MAP_5,
)
from GGanalysis.games.neverness_to_everness.gacha_kernel import (
    ProbabilityMap,
    build_nte_analysis,
    build_cycle_kernel,
    build_nte_transition,
)
from GGanalysis.games.neverness_to_everness.stationary_statistic import (
    NacupedaStatistics,
    NTEStationaryAnalysis,
    arc_disc_stationary_rates,
)
from GGanalysis.games.neverness_to_everness.cosmetic_model import NTECosmeticModel


__all__ = [
    "build_nte_analysis",
    "build_nte_transition",
    "build_cycle_kernel",
    "NacupedaStatistics",
    "NTEStationaryAnalysis",
    "NTEMonopolyModel",
    "NTECosmeticModel",
    "character_stationary",
    "standard_stationary",
    "up_5star_character",
    "standard_5star_character",
    "PITY_ARC_DISC_S",
    "NTEArcDiscUPModel",
    "up_5star_arc_disc",
    "arc_disc_stationary_rates",
]


class NTEMonopolyModel(GachaModel):
    """由一组棋盘配置定义的异环 S 级角色模型。"""

    def __init__(
        self,
        normal_map_info: Sequence[str],
        pity_map_info: Sequence[str],
        pity_start: int,
        normal_probability_map: ProbabilityMap,
        pity_probability_map: ProbabilityMap,
    ) -> None:
        super().__init__()
        self.stationary = NTEStationaryAnalysis(
            normal_map_info,
            pity_map_info,
            pity_start,
            normal_probability_map,
            pity_probability_map,
        )

    @property
    def tm(self) -> MarkovTransition:
        return self.stationary.transition

    @property
    def ss(self) -> StateSpace:
        return self.stationary.hit_analysis.process.full_space

    @property
    def kernel(self) -> StateKernel:
        return self.stationary.kernel

    def __call__(
        self,
        item_num: int = 1,
        multi_dist: bool = False,
        start_pos: int = 0,
        item_pity: int = 0,
        method: ConvolutionMethod = "auto",
        strategy: str = "step",
    ) -> Union[FiniteDist, list[FiniteDist]]:
        """计算获得 item_num 个 S 的花费；strategy 显式选择 step/power。

        multi_dist=True 始终顺序产生所有件数结果；method 控制卷积后端。
        """
        if strategy not in ("step", "power"):
            raise ValueError("strategy must be step or power.")
        if not isinstance(item_num, int) or item_num < 0:
            raise ValueError("item_num must be a non-negative integer.")
        if item_num == 0:
            return FiniteDist.delta(0)

        analysis = self.stationary.hit_analysis
        initial = analysis.process.boundary_space.delta([start_pos])
        if not multi_dist:
            return analysis.nth_state_dist(
                item_num, initial, initial_layer=item_pity, method=method, strategy=strategy,
            ).marginal_cost()
        return [FiniteDist.delta(0)] + [
            joint.marginal_cost() for joint in analysis.iter_state_dists(
                item_num, initial, initial_layer=item_pity, method=method,
            )
        ]

    def item_num_dist(self, pull: int, start_pos: int = 0, item_pity: int = 0,
                      multi_dist: bool = False, *, method: ConvolutionMethod = "auto"
                      ) -> FiniteDist | list[FiniteDist]:
        """返回投入 pull 抽的 S 数量分布；multi_dist 遍历 0 至 pull 抽。"""
        analysis = self.stationary.hit_analysis
        initial = analysis.process.boundary_space.delta([start_pos])
        return analysis.item_num_dist(pull, initial, item_pity, multi_dist, method=method)

    @property
    def stationary_probability(self) -> float:
        """长期每抽获得 S 的概率。"""
        return self.stationary.hit_analysis.long_run_hit_rate

    @property
    def nacupeda_statistics(self) -> NacupedaStatistics:
        """当前 banner 的纳库佩达事件统计。"""
        return self.stationary.nacupeda_statistics

    @cached_property
    def _cosmetics(self) -> dict[str, NTECosmeticModel]:
        tiles = {tile.split("_")[0] for tile in
                 (*self.stationary.normal_map_info, *self.stationary.pity_map_info)}
        return {name: NTECosmeticModel(self.stationary, tile, cap, duplicate)
                for name, (tile, cap, duplicate) in CHARACTER_COSMETIC_RULES.items() if tile in tiles}

    @property
    def cosmetics(self) -> dict[str, NTECosmeticModel]:
        """当前棋盘的 wing/paint/outfit 装扮模型；不含对应格子的棋盘返回空字典。"""
        return dict(self._cosmetics)

class NTEArcDiscUPModel(PityBernoulliModel):
    """逐件判定的 UP S 弧盘花费模型。

    任意 S 清 S 计数，只有 UP 清 UP 计数；80 抽保底强制 UP。
    UP 后 S/UP 状态归零，继承的 A 计数不影响 UP，因此后续周期 IID。
    花费单位为一件弧盘，不对十连支付取整。
    """

    def __init__(self) -> None:
        super().__init__(PITY_ARC_DISC_S, ARC_DISC_UP_GIVEN_S)

    def _first_dist(self, item_pity: int = 0, up_pity: int = 0) -> FiniteDist:
        cap = ARC_DISC_UP_PITY - up_pity
        dist = super().__call__(item_pity=item_pity).dist
        return FiniteDist(np.r_[dist[:cap], 1 - dist[:cap].sum()])

    def __call__(self, item_num: int = 1, multi_dist: bool = False,
                 item_pity: int = 0, up_pity: int = 0) -> FiniteDist | list[FiniteDist]:
        """返回获得指定数量 UP S 所需的累计件数分布。

        Parameters
        ----------
        item_num : int
            目标 UP S 数量。
        multi_dist : bool
            为真时返回从零件到 item_num 件的累计花费分布列表。
        item_pity : int
            当前连续未出任意 S 的件数，仅影响首个 UP 周期。
        up_pity : int
            当前连续未出 UP S 的件数，仅影响首个 UP 周期。

        Returns
        -------
        FiniteDist or list[FiniteDist]
            索引为逐件花费；十连支付需对累计件数整体向上取整到十的倍数。
        """
        answers = [FiniteDist.delta(0)]
        if item_num:
            answers.append(self._first_dist(item_pity, up_pity))
            if item_num > 1:
                cycle_dist = self._first_dist()
                for _ in range(1, item_num):
                    answers.append(answers[-1] * cycle_dist)
        return answers if multi_dist else answers[-1]

up_5star_character = NTEMonopolyModel(
    CHARACTER_MAP_INFO,
    CHARACTER_MAP_PITY_INFO,
    PITY_START,
    CHARACTER_POOL_NORMAL_MAP_5,
    CHARACTER_POOL_PITY_MAP_5,
)
standard_5star_character = NTEMonopolyModel(
    STANDARD_MAP_INFO,
    STANDARD_MAP_PITY_INFO,
    PITY_START,
    STANDARD_POOL_NORMAL_MAP_5,
    STANDARD_POOL_PITY_MAP_5,
)

character_stationary: NTEStationaryAnalysis = up_5star_character.stationary
standard_stationary: NTEStationaryAnalysis = standard_5star_character.stationary


# 弧盘研募：任意 S 基础概率 3%，60 抽保 S；每次出 S 有 25% 为当期 UP。
PITY_ARC_DISC_S = np.zeros(ARC_DISC_S_PITY + 1)
PITY_ARC_DISC_S[1:ARC_DISC_S_PITY] = ARC_DISC_S_PROBABILITY
PITY_ARC_DISC_S[ARC_DISC_S_PITY] = 1

up_5star_arc_disc = NTEArcDiscUPModel()


if __name__ == "__main__":
    stationary = character_stationary
    roll_probability = float(stationary.position_entry_probabilities.sum())
    tile_probabilities = stationary.tile_entry_probabilities
    tile_reward_total = sum(
        probability for tile, probability in tile_probabilities.items() if tile != "C"
    )
    nacupeda = stationary.nacupeda_statistics

    print("=== 异环角色池模型检查 ===")
    print(f"平稳单 S 期望: {stationary.steady_dist.exp:.6f}")
    print(f"长期 S 概率: {up_5star_character.stationary_probability:.6%}")
    print(f"每抽实际掷骰概率: {roll_probability:.9%}")
    print(f"第 90 抽硬保底原地不动概率: {1 - roll_probability:.9%}")

    print("\n=== 各类格子的平稳进入概率 ===")
    print("格子       每抽无条件概率    实际掷骰条件概率")
    for tile, probability in tile_probabilities.items():
        suffix = "（附加事件）" if tile == "C" else ""
        print(
            f"{tile:<8} {probability:>14.9%} "
            f"{probability / roll_probability:>17.9%} {suffix}"
        )
    print(f"互斥奖励概率和（不含 C）: {tile_reward_total:.15f}")
    print(f"与实际掷骰概率之差: {tile_reward_total - roll_probability:+.3e}")

    print("\n=== 纳库佩达沉眠地与守护者追逐 ===")
    print(f"进入 C 事件格的每抽概率: {nacupeda.trigger_probability_per_pull:.9%}")
    print(f"C 事件格平均重访周期: {nacupeda.revisit_period_pulls:.6f} 抽")
    print(f"三次掷骰内追上的概率: {nacupeda.catch_probability_per_trigger:.9%}")
    print(f"每抽触发并追上的概率: {nacupeda.catch_probability_per_pull:.9%}")
    print(
        "平均成功追上一次所需抽数: "
        f"{nacupeda.successful_revisit_period_pulls:.6f} 抽"
    )
