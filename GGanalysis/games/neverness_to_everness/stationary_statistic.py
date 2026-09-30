"""异环棋盘的游戏事件定义；公共分析负责稳态和奖励期望计算。"""

from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from GGanalysis.distribution_1d import FiniteDist
from GGanalysis.markov import HitProcessAnalysis, MarkovTransition, StateRewards, group_mass
from GGanalysis.state_distribution import StateDist, StateKernel
from GGanalysis.games.neverness_to_everness.gacha_data import END_POS, HARD_PITY, parse_tile_type
from GGanalysis.games.neverness_to_everness.gacha_kernel import (
    ProbabilityMap, build_nte_analysis, _position_move_matrix,
)

__all__ = ["NacupedaStatistics", "NTEStationaryAnalysis"]


@dataclass(frozen=True)
class NacupedaStatistics:
    """沉眠地事件的长期率与平均事件间隔，不是任意初态首次等待时间。"""

    trigger_probability_per_pull: float
    revisit_period_pulls: float
    catch_probability_per_trigger: float
    catch_probability_per_pull: float
    successful_revisit_period_pulls: float


@dataclass(frozen=True)
class NTEStationaryAnalysis:
    """固定一组棋盘配置，声明游戏事件并调用公共分析。

    两张 map_info 决定格子标签，两张 probability_map 决定命中机制；
    pity_start 决定切换时刻。配置转为不可变元组，变更规则需创建新对象。
    对外数组、字典和分布均为独立结果，只有私有派生数据缓存。
    """

    normal_map_info: Sequence[str]
    pity_map_info: Sequence[str]
    pity_start: int
    normal_probability_map: ProbabilityMap
    pity_probability_map: ProbabilityMap

    def __post_init__(self) -> None:
        for name in ("normal_map_info", "pity_map_info", "normal_probability_map", "pity_probability_map"):
            values = tuple(getattr(self, name))
            if len(values) != END_POS:
                raise ValueError(f"{name} must have END_POS entries.")
            object.__setattr__(self, name, values)

    @property
    def hit_analysis(self) -> HitProcessAnalysis:
        """同配置共享的公共分析，与初态无关。"""
        return build_nte_analysis(self.normal_probability_map,
                                  self.pity_probability_map, self.pity_start)

    @property
    def transition(self) -> MarkovTransition:
        """当前棋盘的逐抽转移副本。"""
        return self.hit_analysis.process.transition

    @property
    def pity_layer_matrices(self) -> tuple[tuple[sp.csr_matrix, ...], tuple[sp.csr_matrix, ...]]:
        """每层命中、未命中矩阵的副本。"""
        return self.hit_analysis.process.layer_matrices

    @property
    def kernel(self) -> StateKernel:
        """相邻 S 命中间周期核的副本。"""
        return self.hit_analysis.kernel

    def first_state_dist(self, init_pos: int, init_pity: int) -> StateDist:
        """指定位置、垫抽的首件分布；共享周期分析但不缓存任意初态。"""
        analysis = self.hit_analysis
        initial = analysis.process.boundary_space.delta([init_pos])
        return analysis.first_state_dist(initial, initial_layer=init_pity)

    @property
    def post_hit_position_dist(self) -> FiniteDist:
        """命中时观察到的位置稳态分布。"""
        return FiniteDist(self.hit_analysis.post_hit_stationary, trim_tail_zeros=False)

    @property
    def state_array(self) -> np.ndarray:
        """逐抽开始前的 [保底, 位置] 稳态联合分布副本。"""
        return self.hit_analysis.steady_full_state.reshape(HARD_PITY, END_POS)

    @property
    def state_dist(self) -> FiniteDist:
        """展平完整状态编号的逐抽稳态分布。"""
        return FiniteDist(self.state_array.ravel(), trim_tail_zeros=False)

    @property
    def position_dist(self) -> FiniteDist:
        """随机一抽开始前的位置边缘分布。"""
        return FiniteDist(self.state_array.sum(axis=0), trim_tail_zeros=False)

    @property
    def pity_dist(self) -> FiniteDist:
        """随机一抽开始前的保底计数边缘分布。"""
        return FiniteDist(self.state_array.sum(axis=1), trim_tail_zeros=False)

    @property
    def position_move_matrix(self) -> sp.csr_matrix:
        """一次实际掷骰的位置转移矩阵副本。"""
        return _position_move_matrix().copy()

    @cached_property
    def _entry_by_pity(self) -> np.ndarray:
        entry = np.zeros((HARD_PITY, END_POS))
        entry[:-1] = (_position_move_matrix() @ self.state_array[:-1].T).T
        return entry

    @property
    def position_entry_by_pity(self) -> np.ndarray:
        """按起始保底层分组的落点质量；硬保底不掷骰，最后一行为零。"""
        return self._entry_by_pity.copy()

    @property
    def position_entry_probabilities(self) -> np.ndarray:
        """每抽实际进入各位置的无条件质量，独立只读数组。"""
        probabilities = self._entry_by_pity.sum(axis=0)
        probabilities.flags.writeable = False
        return probabilities

    @property
    def position_entry_dist(self) -> FiniteDist:
        """以本抽实际掷骰为条件的落点分布。"""
        probabilities = self.position_entry_probabilities
        return FiniteDist(probabilities / self._event_rates["roll"], trim_tail_zeros=False)

    @cached_property
    def _event_rewards(self) -> StateRewards:
        # 游戏只声明标签与事件条件；每行权重是从完整状态出发的一步事件概率。
        maps = (self.normal_map_info, self.pity_map_info)
        rewards, markers = [], []
        for info in maps:
            fields = [tile.split("_") for tile in info]
            rewards.append(np.array(["_".join(x for x in parts if x != "C") for parts in fields]))
            markers.append(np.array(["C" in parts[1:] for parts in fields], dtype=float))
        names = sorted(set(rewards[0]) | set(rewards[1]))
        weights = {name: np.zeros((HARD_PITY, END_POS)) for name in [*names, "C", "roll"]}
        # 先验证/取得配置，避免无效切换阈值进入切片逻辑。
        analysis = self.hit_analysis
        split = min(self.pity_start, HARD_PITY - 1)
        move = _position_move_matrix()
        for stage, rows in enumerate((slice(0, split), slice(split, HARD_PITY - 1))):
            for name in names:
                weights[name][rows] = move.T @ (rewards[stage] == name).astype(float)
            weights["C"][rows] = move.T @ markers[stage]
        weights["roll"][:-1] = 1
        return StateRewards(analysis.process.full_space,
                            {name: values.ravel() for name, values in weights.items()})

    @property
    def event_rewards(self) -> StateRewards:
        """声明的格子、C 事件及掷骰权重；StateRewards 的公开数据均为副本。"""
        return self._event_rewards

    @cached_property
    def _event_rates(self) -> dict[str, float]:
        return self._event_rewards.expectation(self.hit_analysis.steady_full_state)

    @property
    def tile_entry_probabilities(self) -> dict[str, float]:
        """每抽奖励格和附加 C 事件的无条件概率；C 与格子奖励可以重叠。"""
        return {name: rate for name, rate in self._event_rates.items() if name != "roll"}

    @property
    def tile_type_probabilities(self) -> dict[str, float]:
        """实际掷骰条件下的基础格子类型概率。"""
        rates = {name: value for name, value in self.tile_entry_probabilities.items() if name != "C"}
        grouped = group_mass(list(rates.values()), [parse_tile_type(name) for name in rates])
        roll_rate = self._event_rates["roll"]
        if roll_rate <= 0:
            raise ValueError("conditional tile probabilities require a positive roll rate.")
        return {name: rate / roll_rate for name, rate in sorted(grouped.items())}

    @cached_property
    def nacupeda_statistics(self) -> NacupedaStatistics:
        """沉眠地事件统计；复用 C 触发率，只保留游戏专用的三骰追逐条件。"""
        trigger_probability = self.tile_entry_probabilities["C"]
        catch_probability = sum(
            d1 + d2 >= 11 or d1 + d2 + d3 >= 13
            for d1, d2, d3 in product(range(1, 7), repeat=3)
        ) / 6 ** 3
        success_rate = trigger_probability * catch_probability
        return NacupedaStatistics(
            trigger_probability, 1 / trigger_probability if trigger_probability else float("inf"),
            catch_probability, success_rate, 1 / success_rate if success_rate else float("inf"),
        )

    @property
    def steady_dist(self) -> FiniteDist:
        """稳态条件下相邻 S 命中的花费分布副本。"""
        return self.hit_analysis.steady_dist
