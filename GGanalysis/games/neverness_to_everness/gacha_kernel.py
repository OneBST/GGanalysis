"""异环棋盘抽卡模型的转移矩阵与首次命中计算核心。"""

from functools import lru_cache
from typing import Iterator

import numpy as np
import scipy.sparse as sp

from GGanalysis.markov import (
    MarkovTransition, StateSpace, ProbabilityMatrixBuilder, HitProcess, HitProcessAnalysis,
)
from GGanalysis.state_distribution import StateKernel
from GGanalysis.games.neverness_to_everness.gacha_data import END_POS, HARD_PITY


ProbabilityMap = tuple[float, ...]

__all__ = [
    "ProbabilityMap",
    "build_nte_analysis",
    "build_nte_transition",
    "build_cycle_kernel",
]


@lru_cache(maxsize=1)
def _position_move_matrix() -> sp.csr_matrix:
    """一次掷骰的位置转移；规则遍历和边合并由公共构建器负责。

    私有缓存只供内部读取，公开访问由调用方返回副本。
    """
    def rule(state: np.ndarray) -> Iterator[tuple[list[int], float]]:
        pos = int(state[0])
        for dice in range(1, 7):
            if pos == 16:
                end = 54 + dice
            elif pos == 43:
                end = 63 + dice
            elif 55 <= pos <= 63:
                end = pos + dice if pos + dice <= 63 else pos + dice - 64 + 18
            elif 64 <= pos <= 72:
                end = pos + dice if pos + dice <= 72 else pos + dice - 73 + 45
            else:
                end = (pos + dice - 1) % 54 + 1
            yield [end], 1 / 6

    space = StateSpace.from_shape([END_POS])
    builder = ProbabilityMatrixBuilder(space, space)
    builder.add_rule(rule)
    return builder.build()


@lru_cache(maxsize=8)
def _get_nte_analysis(
    normal_probability_map: ProbabilityMap,
    pity_probability_map: ProbabilityMap,
    pity_start: int,
) -> HitProcessAnalysis:
    """声明普通、变格及原地硬保底三类规则，由公共工具组装完整过程。"""
    move = _position_move_matrix()
    stages = []
    for probabilities in (normal_probability_map, pity_probability_map):
        probabilities = np.asarray(probabilities, dtype=float)
        if (probabilities.shape != (END_POS,) or not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0) or np.any(probabilities > 1)):
            raise ValueError("probability maps must contain END_POS probabilities in [0, 1].")
        # 落点决定是否命中，因此按行乘概率；移动规则只构造一次。
        stages.append((move.multiply(probabilities[:, None]).tocsr(),
                       move.multiply((1 - probabilities)[:, None]).tocsr()))
    layers = [stages[int(pity >= pity_start)] for pity in range(HARD_PITY - 1)]
    # 第 90 抽原地必成，不掷骰；不能沿用普通阶段的移动矩阵。
    layers.append((sp.eye(END_POS, format="csr"), sp.csr_matrix((END_POS, END_POS))))
    hits, misses = zip(*layers)
    process = HitProcess.from_layers(
        StateSpace.from_shape([HARD_PITY, END_POS]), StateSpace.from_shape([END_POS]),
        hits, misses, embed=np.arange(END_POS),
    )
    return HitProcessAnalysis(process, HARD_PITY)


def build_nte_analysis(
    normal_probability_map: ProbabilityMap,
    pity_probability_map: ProbabilityMap,
    pity_start: int,
) -> HitProcessAnalysis:
    """构造并复用与初态无关的棋盘分析。

    两张长度为 END_POS 的概率图及变格阈值定义配置；同配置共享有界缓存，
    不绑定初始位置或垫抽。公共矩阵、核及稳态结果由分析对象返回独立副本。
    """
    # 在缓存查找前验证，避免 True、1.0 与整数 1 的相等键绕过检查。
    if (isinstance(pity_start, (bool, np.bool_)) or not isinstance(pity_start, (int, np.integer))
            or not 0 <= pity_start <= HARD_PITY):
        raise ValueError("pity_start must be an integer in [0, HARD_PITY].")
    return _get_nte_analysis(tuple(normal_probability_map), tuple(pity_probability_map), int(pity_start))


def build_nte_transition(
    normal_probability_map: ProbabilityMap,
    pity_probability_map: ProbabilityMap,
    pity_start: int,
) -> MarkovTransition:
    """返回完整逐抽转移的独立副本，修改结果不污染同配置缓存。"""
    return build_nte_analysis(normal_probability_map, pity_probability_map, pity_start).process.transition


def build_cycle_kernel(
    normal_probability_map: ProbabilityMap,
    pity_probability_map: ProbabilityMap,
    pity_start: int,
) -> StateKernel:
    """返回相邻 S 命中的周期核副本，系数为 ``[抽数, 结束位置, 开始位置]``。

    获得 S 后保底归零而位置继承；核保留 73 个位置状态。
    周期和长期统计共用 ``build_nte_analysis`` 内部缓存。
    """
    return build_nte_analysis(normal_probability_map, pity_probability_map, pity_start).kernel
