"""异环棋盘抽卡模型的转移矩阵与首次命中计算核心。"""

from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from GGanalysis.markov import MarkovTransition, StateSpace, HitTransitionBuilder
from GGanalysis.state_distribution import StateDist, StateKernel
from GGanalysis.games.neverness_to_everness.gacha_data import END_POS, HARD_PITY


ProbabilityMap = tuple[float, ...]

__all__ = [
    "ProbabilityMap",
    "build_nte_transition",
    "build_cycle_kernel",
]


@lru_cache
def _get_next_positions(pos: int) -> tuple[tuple[int, float], ...]:
    """返回当前位置掷骰后所有可能的下一位置及其概率。"""
    # 进入右侧特殊区域
    if pos == 16:
        return tuple((54 + dice, 1 / 6) for dice in range(1, 7))
    # 进入左侧特殊区域
    if pos == 43:
        return tuple((63 + dice, 1 / 6) for dice in range(1, 7))
    # 右侧特殊区域内部
    if 55 <= pos <= 63:
        return tuple(
            (pos + dice if pos + dice <= 63 else pos + dice - 64 + 18, 1 / 6)
            for dice in range(1, 7)
        )
    # 左侧特殊区域内部
    if 64 <= pos <= 72:
        return tuple(
            (pos + dice if pos + dice <= 72 else pos + dice - 73 + 45, 1 / 6)
            for dice in range(1, 7)
        )
    return tuple(((pos + dice - 1) % 54 + 1, 1 / 6) for dice in range(1, 7))


@lru_cache(maxsize=8)
def _build_nte_process(
    normal_probability_map: ProbabilityMap,
    pity_probability_map: ProbabilityMap,
    pity_start: int,
):
    """逐条标记棋盘的命中与未命中边，并按位置保留命中后状态。"""
    space = StateSpace.from_shape([HARD_PITY, END_POS])
    boundary = StateSpace.from_shape([END_POS])
    builder = HitTransitionBuilder(space, boundary, np.arange(END_POS))

    miss_from: list[int] = []
    miss_to: list[int] = []
    miss_prob: list[float] = []
    hit_from: list[int] = []
    hit_to: list[int] = []
    hit_prob: list[float] = []

    for pity in range(HARD_PITY):
        for pos in range(END_POS):
            from_id = pity * END_POS + pos
            # 90保底特判
            if pity == HARD_PITY - 1:
                hit_from.append(from_id)
                hit_to.append(pos)
                hit_prob.append(1.0)
                continue

            probability_map = (
                pity_probability_map
                if pity >= pity_start
                else normal_probability_map
            )
            for next_pos, dice_probability in _get_next_positions(pos):
                obtain_probability = probability_map[next_pos]
                if obtain_probability > 0:
                    hit_from.append(from_id)
                    hit_to.append(next_pos)
                    hit_prob.append(obtain_probability * dice_probability)
                if obtain_probability < 1:
                    miss_from.append(from_id)
                    miss_to.append((pity + 1) * END_POS + next_pos)
                    miss_prob.append((1 - obtain_probability) * dice_probability)

    builder.add_miss_many(miss_from, miss_to, miss_prob)
    builder.add_hit_many(hit_from, hit_to, hit_prob)
    return builder.build(layered=True)


@lru_cache(maxsize=8)
def build_nte_transition(
    normal_probability_map: ProbabilityMap,
    pity_probability_map: ProbabilityMap,
    pity_start: int,
) -> MarkovTransition:
    """构造 ``[pity, position]`` 上的逐抽稀疏转移矩阵。"""
    return _build_nte_process(
        normal_probability_map, pity_probability_map, pity_start
    ).transition


@lru_cache(maxsize=8)
def _get_pity_layer_matrices(
    normal_probability_map: ProbabilityMap,
    pity_probability_map: ProbabilityMap,
    pity_start: int,
) -> tuple[tuple[sp.csr_matrix, ...], tuple[sp.csr_matrix, ...]]:
    """从命中过程提取每个 pity 层的命中与未命中矩阵。"""
    return _build_nte_process(
        normal_probability_map, pity_probability_map, pity_start
    ).layer_matrices


@lru_cache(maxsize=8)
def build_cycle_kernel(
    normal_probability_map: ProbabilityMap,
    pity_probability_map: ProbabilityMap,
    pity_start: int,
) -> StateKernel:
    """构造获得一个 S 后到再次获得 S 的周期核。

    获得 S 后 pity 重置为 0，但棋盘停留在本次命中的位置。
    返回核满足
    ``kernel[t, end_pos, start_pos]``：刚在 ``start_pos`` 获得 S 后，
    再花费 ``t`` 抽于 ``end_pos`` 获得下一个 S 的概率。

    任意 ``init_pos``、``init_pity`` 下的首件分布由同一个命中过程计算。
    """
    kernel = _build_nte_process(
        normal_probability_map, pity_probability_map, pity_start
    ).cycle_kernel(HARD_PITY)
    kernel.validate(atol=1e-11)
    return kernel


@lru_cache(maxsize=32)
def _get_first_state_dist(
    init_pos: int,
    init_pity: int,
    normal_probability_map: ProbabilityMap,
    pity_probability_map: ProbabilityMap,
    pity_start: int,
) -> StateDist:
    """构造指定位置和 pity 下首个 S 的花费—落点联合分布。"""
    if not 0 <= init_pos < END_POS:
        raise ValueError(f"init_pos must be in [0, {END_POS - 1}].")
    if not 0 <= init_pity < HARD_PITY:
        raise ValueError(f"init_pity must be in [0, {HARD_PITY - 1}].")
    initial = np.zeros(END_POS, dtype=np.float64)
    initial[init_pos] = 1.0
    return _build_nte_process(
        normal_probability_map, pity_probability_map, pity_start
    ).first_hit(initial, HARD_PITY - init_pity, initial_layer=init_pity)
