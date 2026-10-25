"""异环棋盘的游戏事件定义；公共分析负责稳态和奖励期望计算。"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import product
from typing import Sequence

import numpy as np
import scipy.sparse as sp
from scipy.stats import binom

from GGanalysis.distribution_1d import FiniteDist
from GGanalysis.markov import (
    HitProcessAnalysis, MarkovTransition, StateRewards, StateSpace,
    TransitionBuilder, group_mass, stationary_power,
)
from GGanalysis.state_distribution import StateDist, StateKernel
from GGanalysis.games.neverness_to_everness.gacha_data import END_POS, HARD_PITY, parse_tile_type
from GGanalysis.games.neverness_to_everness.gacha_data import (
    ARC_DISC_S_PROBABILITY, ARC_DISC_UP_GIVEN_S, ARC_DISC_A_PROBABILITY,
    ARC_DISC_S_PITY, ARC_DISC_UP_PITY, ARC_DISC_A_PITY,
    POINT_GIFT_PERIOD, POINT_GIFT_A_CHARACTER_PROBABILITY,
    POINT_GIFT_A_ARC_DISC_PROBABILITY,
)
from GGanalysis.games.neverness_to_everness.gacha_kernel import (
    ProbabilityMap, build_nte_analysis, _position_move_matrix,
)

__all__ = ["NacupedaStatistics", "NTEStationaryAnalysis", "arc_disc_stationary_rates"]


def _arc_disc_outcomes(state: tuple[int, int, int]
                       ) -> list[tuple[tuple[int, int, int], float, str]]:
    """逐件互斥结果；优先级为 UP80、S60、A10，S 不清 A。"""
    s, u, a = state
    if u == ARC_DISC_UP_PITY - 1:
        probabilities = (1.0, 0.0, 0.0, 0.0)
    else:
        p_s = 1.0 if s == ARC_DISC_S_PITY - 1 else ARC_DISC_S_PROBABILITY
        p_a = 1.0 - p_s if a == ARC_DISC_A_PITY - 1 else min(ARC_DISC_A_PROBABILITY, 1.0 - p_s)
        probabilities = (p_s * ARC_DISC_UP_GIVEN_S,
                         p_s * (1.0 - ARC_DISC_UP_GIVEN_S),
                         p_a, 1.0 - p_s - p_a)
    next_a = min(a + 1, ARC_DISC_A_PITY - 1)
    ends = ((0, 0, next_a), (0, u + 1, next_a),
            (s + 1, u + 1, 0), (s + 1, u + 1, next_a))
    return [(end, p, kind) for end, p, kind in
            zip(ends, probabilities, ("UP", "non_UP_S", "A", "B")) if p > 0]


def _build_arc_disc_process() -> tuple[tuple[tuple[int, int, int], ...], MarkovTransition, StateRewards]:
    """从零状态枚举可达联合状态，以公共工具组装稀疏转移和奖励。"""
    states = [(0, 0, 0)]
    lookup = {states[0]: 0}
    sources, targets, probabilities, kinds = [], [], [], []
    # 列表迭代会继续访问末尾新增状态，即广度优先枚举。
    for source, state in enumerate(states):
        for end, probability, kind in _arc_disc_outcomes(state):
            if end not in lookup:
                lookup[end] = len(states)
                states.append(end)
            sources.append(source)
            targets.append(lookup[end])
            probabilities.append(probability)
            kinds.append(kind)
    space = StateSpace.from_shape([len(states)])
    builder = TransitionBuilder(space)
    builder.add_many(sources, targets, probabilities)
    transition = builder.build(check=True)
    weights = {name: np.zeros(len(states)) for name in ("S", "UP", "A", "B")}
    for source, probability, kind in zip(sources, probabilities, kinds):
        if kind in ("UP", "non_UP_S"):
            weights["S"][source] += probability
        if kind != "non_UP_S":
            weights[kind][source] += probability
    return tuple(states), transition, StateRewards(space, weights)


@lru_cache(maxsize=1)
def _arc_disc_stationary_rates() -> tuple[tuple[str, float], ...]:
    states, transition, rewards = _build_arc_disc_process()
    initial = np.zeros(len(states))
    initial[0] = 1.0
    steady, info = stationary_power(transition, x0=initial, tol=1e-13, return_info=True)
    if not info.converged:
        raise RuntimeError("弧盘联合保底链的平稳分布未收敛。")
    return tuple(rewards.expectation(steady).items())


def arc_disc_stationary_rates() -> dict[str, float]:
    """返回考虑 S/UP/A 保底的长期每件综合概率。

    Returns
    -------
    dict[str, float]
        ``S`` 为任意 S（包含 UP），``UP`` 为 UP S，``A``、``B`` 为对应品质。
        互斥品质满足 S + A + B = 1，UP 是 S 的子集，不能再相加。
        每次调用返回独立字典，首次调用计算并缓存长期率。

    Notes
    -----
    采用逐件规则：基础 S 为 3%，其中 UP 占 25%，基础 A 为 7%；60 件保 S、
    80 件保 UP、10 件保 A。优先判 UP，再判 S，最后判 A；A级保底时先保留
    S 概率，剩余质量给 A。UP 清 S/UP，非 UP S 只清 S；所有 S 均不清 A。
    A 计数饱和到 9，实际出 A 才归零。各计数跨期继承，不考虑额外奖励。
    完整状态为逐件抽取前的 (未出 S 件数, 未出 UP 件数, 未出 A 件数)。
    这是长期事件率，不是零进度首件等待的倒数，也不是固定预算下的命中概率。
    """
    return dict(_arc_disc_stationary_rates())


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

    @property
    def a_grade_stationary_rates(self) -> dict[str, float]:
        """角色棋盘每次主抽的 A 级期望数量，包含集点赠礼。

        Returns
        -------
        dict[str, float]
            ``A_character``、``A_arc_disc`` 分别为 A 角色、A 弧盘的期望数量，
            ``A_total`` 为二者之和。落格 A/W 奖励分别加上每十抽赠礼的
            20%/80% 贡献；赠礼独立于落格及 S 进度，因此同抽可以获得两件 A。
            这是等效综合产率，不是每抽至少获得一件 A 的概率。

        Notes
        -----
        主抽包含原地 S 硬保底。累计每十次主抽额外赠送一件 A；自然出 A 或 S
        不重置赠礼计数，不修改棋盘移动与 S 转移。跨限定棋盘继承赠礼计数。
        """
        entries = self.tile_entry_probabilities
        character = entries.get("A", 0.0) + POINT_GIFT_A_CHARACTER_PROBABILITY / POINT_GIFT_PERIOD
        arc_disc = entries.get("W", 0.0) + POINT_GIFT_A_ARC_DISC_PROBABILITY / POINT_GIFT_PERIOD
        return {"A_character": character, "A_arc_disc": arc_disc,
                "A_total": character + arc_disc}

    def point_gift_dist(self, pull: int, gift_pity: int = 0) -> dict[str, FiniteDist]:
        """给定主抽数，返回额外赠礼中两种 A 的数量边缘分布。

        Parameters
        ----------
        pull : int
            接下来投入的主抽数，包含原地 S 硬保底抽。
        gift_pity : int
            已累计但尚未触发下一次赠礼的主抽数，取 0 至 9，与 S 进度无关。

        Returns
        -------
        dict[str, FiniteDist]
            ``A_character`` 和 ``A_arc_disc`` 为赠礼数量边缘分布，不含棋盘奖励。
            赠礼总数 m=(gift_pity+pull)//10；角色数服从 Binomial(m, 0.2)，
            弧盘数等于 m 减角色数。两者相关，不能当作独立分布卷积。
            剩余赠礼进度为 (gift_pity+pull)%10。

        Notes
        -----
        每次赠礼独立按 20%/80% 选择类型；赠礼与棋盘落格独立。
        """
        gifts = (gift_pity + pull) // POINT_GIFT_PERIOD
        character = binom.pmf(np.arange(gifts + 1), gifts, POINT_GIFT_A_CHARACTER_PROBABILITY)
        return {"A_character": FiniteDist(character),
                "A_arc_disc": FiniteDist(character[::-1])}

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
