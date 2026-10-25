"""角色棋盘装扮：有限领取期限的首次获得及长期实际落格率。"""

from functools import cached_property

import numpy as np
import scipy.sparse as sp

from GGanalysis.basic_models import GachaModel
from GGanalysis.distribution_1d import FiniteDist
from GGanalysis.markov import MarkovTransition, StateSpace, first_hitting_time
from GGanalysis.games.neverness_to_everness.gacha_data import (
    END_POS, HARD_PITY, parse_tile_type,
)
from GGanalysis.games.neverness_to_everness.stationary_statistic import NTEStationaryAnalysis


__all__ = ["NTECosmeticModel"]


class NTECosmeticModel(GachaModel):
    """组合角色棋盘转移与一次性累计领取规则的装扮模型。

    Parameters
    ----------
    stationary : NTEStationaryAnalysis
        所属棋盘的共享分析对象，提供转移及完整逐抽稳态。
    tile : str
        目标装扮格标签，如 HD3、HD2、HD1。
    claim_at : int
        同名棋盘从累计零开始可领取装扮的主抽次数，只领取一次，不循环重置。
    duplicate_reward : int
        重复装扮兑换的失纬棋子数，不改变首次获得或棋盘运动。

    Notes
    -----
    首获查询假设尚未拥有目标装扮，并在达到领取期限时立即领取。
    所有花费为主抽，含原地 S 硬保底；硬保底不领取所在格子的奖励。
    累计装扮进度与 S 进度独立，同名棋盘继承累计进度。
    长期查询只统计实际落格；一次性领取对无限长期率的贡献为零。
    """

    def __init__(self, stationary: NTEStationaryAnalysis, tile: str,
                 claim_at: int, duplicate_reward: int) -> None:
        self.stationary = stationary
        self.tile = tile
        self.claim_at = claim_at
        self.duplicate_reward = duplicate_reward

    @cached_property
    def _first_hit_transition(self) -> MarkovTransition:
        """沿用完整棋盘转移，将实际获奖的边汇入一个额外吸收态。"""
        process = self.stationary.hit_analysis.process
        full = process.transition.P.tocsr()
        size = process.full_space.N
        edges = full.tocoo()
        maps = (self.stationary.normal_map_info, self.stationary.pity_map_info)
        target = np.array([[parse_tile_type(t) == self.tile for t in info] for info in maps])
        layers = edges.col // END_POS
        stage = (layers >= self.stationary.pity_start).astype(int)
        selected = target[stage, edges.row % END_POS] & (layers < HARD_PITY - 1)
        hit = sp.csr_matrix((edges.data[selected], (edges.row[selected], edges.col[selected])),
                            shape=full.shape)
        miss = full - hit
        hit_probability = sp.csr_matrix(np.asarray(hit.sum(axis=0)))
        matrix = sp.bmat([[miss, sp.csr_matrix((size, 1))],
                          [hit_probability, sp.csr_matrix([[1.0]])]], format="csr")
        transition = MarkovTransition(StateSpace.from_shape([size + 1]), matrix)
        transition.validate()
        return transition

    def _first_dist(self, initial: np.ndarray, cosmetic_pity: int) -> FiniteDist:
        remaining = max(0, self.claim_at - cosmetic_pity)
        if remaining == 0:
            return FiniteDist.delta(0)
        size = self.stationary.hit_analysis.process.full_space.N
        # 只算领取期限之前的自然首达；最后一抽全部剩余路径领取，不存在截断尾部。
        pmf, unclaimed = first_hitting_time(
            self._first_hit_transition, np.r_[initial, 0.0], [size], remaining - 1,
        )
        return FiniteDist(np.r_[pmf, unclaimed], trim_tail_zeros=False, tail_mass=0.0)

    def __call__(self, start_pos: int = 0, item_pity: int = 0,
                 cosmetic_pity: int = 0) -> FiniteDist:
        """给定位置和 S 进度，从当前状态到首次获得装扮的主抽分布。

        Parameters
        ----------
        start_pos : int
            当前棋盘位置，默认初始位置 0。
        item_pity : int
            当前连续未获得 S 的主抽数，默认 0。
        cosmetic_pity : int
            同名棋盘已累计的装扮领取主抽次数，默认 0；自然获得 S 不重置。

        Returns
        -------
        FiniteDist
            含一次性累计领取的完整首获分布；索引为还需投入的主抽数。
            ``exp`` 为期望，``tail_mass`` 为零。
        """
        initial = self.stationary.hit_analysis.process.full_space.delta([item_pity, start_pos])
        return self._first_dist(initial, cosmetic_pity)

    def stationary_start_dist(self, cosmetic_pity: int = 0) -> FiniteDist:
        """从长期刚出 S 后的位置稳态开始，计算含累计领取的首次获得分布。

        Parameters
        ----------
        cosmetic_pity : int
            装扮已累计领取次数，默认从零开始，与位置达到稳态的历史分开指定。

        Returns
        -------
        FiniteDist
            位置按长期刚出 S 后的位置稳态混合，S 进度统一归零，包含领取期限。
            这是重置装扮累计后的首次等待，不是装扮的相邻落格重访分布。
        """
        analysis = self.stationary.hit_analysis
        initial = np.asarray(analysis.process.embed @ analysis.post_hit_stationary).ravel()
        return self._first_dist(initial, cosmetic_pity)

    @property
    def stationary_probability(self) -> float:
        """长期每次主抽实际进入目标装扮格的概率，不含一次性领取。"""
        return self.stationary.tile_entry_probabilities.get(self.tile, 0.0)

    @property
    def revisit_expectation(self) -> float:
        """长期实际落格的平均重访主抽数，直接取进入率倒数，不生成重访分布。"""
        probability = self.stationary_probability
        return 1.0 / probability if probability else float("inf")
