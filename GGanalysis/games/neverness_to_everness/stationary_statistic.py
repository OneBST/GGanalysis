"""异环棋盘 banner 的长期平稳统计。"""

from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from GGanalysis.distribution_1d import FiniteDist
from GGanalysis.markov import HitProcessAnalysis, MarkovTransition
from GGanalysis.state_distribution import StateDist, StateKernel
from GGanalysis.games.neverness_to_everness.gacha_data import (
    END_POS,
    HARD_PITY,
    parse_tile_type,
)
from GGanalysis.games.neverness_to_everness.gacha_kernel import (
    ProbabilityMap,
    _build_nte_process,
    _get_first_state_dist,
    _get_next_positions,
    _get_pity_layer_matrices,
    build_cycle_kernel,
    build_nte_transition,
)


__all__ = ["NacupedaStatistics", "NTEStationaryAnalysis"]


@dataclass(frozen=True)
class NacupedaStatistics:
    """纳库佩达沉眠地事件的长期统计量。"""

    trigger_probability_per_pull: float
    revisit_period_pulls: float
    catch_probability_per_trigger: float
    catch_probability_per_pull: float
    successful_revisit_period_pulls: float


def _has_event_marker(tile_info: str, marker: str) -> bool:
    """检查格子信息下划线后的附加字段中是否包含指定事件标记。"""
    return marker in tile_info.split("_")[1:]


class NTEStationaryAnalysis:
    """单个异环 banner 的惰性平稳分析结果。"""

    def __init__(
        self,
        normal_map_info: Sequence[str],
        pity_map_info: Sequence[str],
        pity_start: int,
        normal_probability_map: ProbabilityMap,
        pity_probability_map: ProbabilityMap,
    ) -> None:
        self.normal_map_info = tuple(normal_map_info)
        self.pity_map_info = tuple(pity_map_info)
        self.pity_start = pity_start
        self.normal_probability_map = normal_probability_map
        self.pity_probability_map = pity_probability_map

    def _map_info_for_pity(self, pity: int) -> tuple[str, ...]:
        """返回指定 pity 实际使用的棋盘信息。"""
        return self.pity_map_info if pity >= self.pity_start else self.normal_map_info

    @cached_property
    def transition(self) -> MarkovTransition:
        """当前 banner 的逐抽转移矩阵。"""
        return build_nte_transition(
            self.normal_probability_map,
            self.pity_probability_map,
            self.pity_start,
        )

    @cached_property
    def pity_layer_matrices(
        self,
    ) -> tuple[tuple[sp.csr_matrix, ...], tuple[sp.csr_matrix, ...]]:
        """当前 banner 的逐 pity 命中与未命中矩阵。"""
        return _get_pity_layer_matrices(
            self.normal_probability_map,
            self.pity_probability_map,
            self.pity_start,
        )

    @cached_property
    def kernel(self) -> StateKernel:
        """当前 banner 相邻两次获得 S 之间的周期核。"""
        return build_cycle_kernel(
            self.normal_probability_map,
            self.pity_probability_map,
            self.pity_start,
        )

    def first_state_dist(self, init_pos: int, init_pity: int) -> StateDist:
        """从指定位置和 pity 开始获得首个 S 的联合分布。"""
        return _get_first_state_dist(
            init_pos,
            init_pity,
            self.normal_probability_map,
            self.pity_probability_map,
            self.pity_start,
        )

    @cached_property
    def hit_analysis(self) -> HitProcessAnalysis:
        """用通用命中过程计算周期与逐抽平稳状态。"""
        process = _build_nte_process(
            self.normal_probability_map,
            self.pity_probability_map,
            self.pity_start,
        )
        initial = np.zeros(END_POS)
        initial[0] = 1
        return HitProcessAnalysis(process, initial, HARD_PITY, initial_layer=0)

    @cached_property
    def post_hit_position_dist(self) -> FiniteDist:
        """获得 S 后落点位置的嵌入链平稳分布。"""
        return FiniteDist(self.hit_analysis.post_hit_stationary, trim_tail_zeros=False)

    @cached_property
    def state_array(self) -> np.ndarray:
        """逐抽 ``[pity, position]`` 联合平稳概率数组。"""
        return self.hit_analysis.steady_full_state.reshape(HARD_PITY, END_POS)

    @cached_property
    def state_dist(self) -> FiniteDist:
        """展平状态编号 ``pity * 73 + position`` 的逐抽平稳分布。"""
        return FiniteDist(self.state_array.reshape(-1), trim_tail_zeros=False)

    @cached_property
    def position_dist(self) -> FiniteDist:
        """随机观察任意一抽时所在棋盘位置的边缘分布。"""
        return FiniteDist(self.state_array.sum(axis=0), trim_tail_zeros=False)

    @cached_property
    def pity_dist(self) -> FiniteDist:
        """随机观察任意一抽时所在保底层数的边缘分布。"""
        return FiniteDist(self.state_array.sum(axis=1), trim_tail_zeros=False)

    @cached_property
    def position_move_matrix(self) -> sp.csr_matrix:
        """一次实际掷骰的位置转移矩阵 ``[进入位置, 当前位置]``。"""
        rows: list[int] = []
        cols: list[int] = []
        probabilities: list[float] = []
        for start_pos in range(END_POS):
            for end_pos, probability in _get_next_positions(start_pos):
                rows.append(end_pos)
                cols.append(start_pos)
                probabilities.append(probability)
        matrix = sp.coo_matrix(
            (probabilities, (rows, cols)),
            shape=(END_POS, END_POS),
            dtype=np.float64,
        ).tocsr()
        matrix.sum_duplicates()
        return matrix

    @cached_property
    def position_entry_by_pity(self) -> np.ndarray:
        """按起始 pity 分组的实际掷骰落点概率质量。"""
        entry = np.zeros_like(self.state_array)
        entry[:HARD_PITY - 1] = (
            self.position_move_matrix @ self.state_array[:HARD_PITY - 1].T
        ).T
        return entry

    @cached_property
    def position_entry_probabilities(self) -> np.ndarray:
        """长期每抽实际掷骰进入各位置的无条件概率质量。"""
        probabilities = self.position_entry_by_pity.sum(axis=0)
        probabilities.flags.writeable = False
        return probabilities

    @cached_property
    def position_entry_dist(self) -> FiniteDist:
        """以本抽实际掷骰为条件时，进入各位置的条件分布。"""
        probabilities = self.position_entry_probabilities
        return FiniteDist(probabilities / probabilities.sum(), trim_tail_zeros=False)

    @cached_property
    def tile_type_probabilities(self) -> dict[str, float]:
        """实际掷骰条件下，长期落到各类格子的条件概率。"""
        type_mass: dict[str, float] = {}
        for pity in range(HARD_PITY - 1):
            map_info = self._map_info_for_pity(pity)
            for pos, probability in enumerate(self.position_entry_by_pity[pity]):
                tile_type = parse_tile_type(map_info[pos])
                type_mass[tile_type] = type_mass.get(tile_type, 0.0) + float(probability)
        total_mass = float(self.position_entry_by_pity.sum())
        return {
            tile_type: mass / total_mass
            for tile_type, mass in sorted(type_mass.items())
        }

    @cached_property
    def tile_entry_probabilities(self) -> dict[str, float]:
        """平稳状态下每抽进入各类格子的无条件概率。"""
        probability_by_reward: dict[str, float] = {}
        for pity in range(HARD_PITY - 1):
            map_info = self._map_info_for_pity(pity)
            for pos, probability in enumerate(self.position_entry_by_pity[pity]):
                fields = map_info[pos].split("_")
                reward = "_".join(field for field in fields if field != "C")
                mass = float(probability)
                probability_by_reward[reward] = (
                    probability_by_reward.get(reward, 0.0) + mass
                )
                if "C" in fields[1:]:
                    probability_by_reward["C"] = (
                        probability_by_reward.get("C", 0.0) + mass
                    )
        return dict(sorted(probability_by_reward.items()))

    @cached_property
    def nacupeda_statistics(self) -> NacupedaStatistics:
        """沉眠地的平稳重访周期及触发后追上守护者的概率。"""
        trigger_probability = 0.0
        for pity in range(HARD_PITY - 1):
            map_info = self._map_info_for_pity(pity)
            trigger_probability += sum(
                float(self.position_entry_by_pity[pity, pos])
                for pos, tile in enumerate(map_info)
                if _has_event_marker(tile, "C")
            )
        outcomes = product(range(1, 7), repeat=3)
        catch_probability = sum(
            dice_1 + dice_2 >= 11 or dice_1 + dice_2 + dice_3 >= 13
            for dice_1, dice_2, dice_3 in outcomes
        ) / (6 ** 3)
        catch_probability_per_pull = trigger_probability * catch_probability
        return NacupedaStatistics(
            trigger_probability_per_pull=trigger_probability,
            revisit_period_pulls=1.0 / trigger_probability,
            catch_probability_per_trigger=catch_probability,
            catch_probability_per_pull=catch_probability_per_pull,
            successful_revisit_period_pulls=1.0 / catch_probability_per_pull,
        )

    @cached_property
    def steady_dist(self) -> FiniteDist:
        """平稳落点条件下，相邻两个 S 之间的抽数分布。"""
        return self.hit_analysis.steady_dist
