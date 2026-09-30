"""异环 Neverness to Everness 测试服棋盘抽卡模型。"""

from typing import Sequence, Union

import numpy as np

from GGanalysis.basic_models import GachaModel, PityBernoulliModel, PityModel
from GGanalysis.distribution_1d import FiniteDist
from GGanalysis.markov import MarkovTransition, StateSpace
from GGanalysis.state_distribution import ConvolutionMethod, StateKernel
from GGanalysis.games.neverness_to_everness.gacha_data import (
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
)


__all__ = [
    "build_nte_analysis",
    "build_nte_transition",
    "build_cycle_kernel",
    "NacupedaStatistics",
    "NTEStationaryAnalysis",
    "NTEMonopolyModel",
    "character_stationary",
    "standard_stationary",
    "up_5star_character",
    "standard_5star_character",
    "PITY_ARC_DISC_S",
    "NTEArcDiscUPModel",
    "first_5star_arc_disc",
    "up_5star_arc_disc",
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
    ) -> Union[FiniteDist, list[FiniteDist]]:
        """计算获得 ``item_num`` 个 S 所需抽数分布。"""
        if not isinstance(item_num, int) or item_num < 0:
            raise ValueError("item_num must be a non-negative integer.")
        if item_num == 0:
            return FiniteDist.delta(0)

        analysis = self.stationary.hit_analysis
        initial = analysis.process.boundary_space.delta([start_pos])
        if not multi_dist:
            return analysis.nth_state_dist(
                item_num, initial, initial_layer=item_pity, method=method,
            ).marginal_cost()
        return [FiniteDist.delta(0)] + [
            joint.marginal_cost() for joint in analysis.iter_state_dists(
                item_num, initial, initial_layer=item_pity, method=method,
            )
        ]

    @property
    def stationary_probability(self) -> float:
        """长期每抽获得 S 的概率。"""
        return self.stationary.hit_analysis.long_run_hit_rate

    @property
    def nacupeda_statistics(self) -> NacupedaStatistics:
        """当前 banner 的纳库佩达事件统计。"""
        return self.stationary.nacupeda_statistics

class NTEArcDiscUPModel(PityBernoulliModel):
    """假设 80 抽保底会重置 S/UP 进度。"""

    def __init__(self):
        super().__init__(PITY_ARC_DISC_S, 0.25)

    def _first_dist(self, item_pity=0, up_pity=0):
        cap = 80 - up_pity
        dist = super().__call__(item_pity=item_pity).dist
        return FiniteDist(np.r_[dist[:cap], 1 - dist[:cap].sum()])

    def __call__(self, item_num=1, multi_dist=False, item_pity=0, up_pity=0):
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
PITY_ARC_DISC_S = np.zeros(61)
PITY_ARC_DISC_S[1:60] = 0.03
PITY_ARC_DISC_S[60] = 1

# 首个任意 S 的分布；多件 S 的联合过程还会受到第 80 抽 UP 保底影响。
first_5star_arc_disc = PityModel(PITY_ARC_DISC_S)
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
