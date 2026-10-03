"""常驻五星角色/武器平稳机制，规则来源于原神抽卡全机制总结。

https://www.bilibili.com/opus/506065155991936609
权重参数是文章模型，不视为官方确认。只计算五星类别，不处理四星耦合。
"""

from functools import cached_property, lru_cache
from typing import Literal

import numpy as np
import scipy.sparse as sp

from GGanalysis.basic_models import GachaModel
from GGanalysis.distribution_1d import FiniteDist
from GGanalysis.markov import StateSpace, ProbabilityMatrixBuilder, HitProcess, HitProcessAnalysis
from GGanalysis.markov.transition import validate_steps

__all__ = ["StandardGenshin5starModel"]

# 类别权重使用本次等待位置 i，即此前未获得次数+1。
_TYPE_BASE_WEIGHT = 30
_TYPE_WEIGHT_START = 146
_TYPE_WEIGHT_STEP = 300
_TYPE_WEIGHT_CEILING = 10000
_TYPE_CERTAIN_PITY = 179


def standard_5star_type_probability(character_pity, weapon_pity) -> np.ndarray:
    """返回本抽已确定为五星时的角色类别概率，支持数组输入。

    两个参数是本抽之前未获得对应类别的抽数，不是包含本抽的等待位置。
    按文章的上限轮盘，较大权重优先；不适用于四星类别。
    """
    character = _TYPE_BASE_WEIGHT + _TYPE_WEIGHT_STEP*np.maximum(
        np.asarray(character_pity) + 1 - _TYPE_WEIGHT_START, 0)
    weapon = _TYPE_BASE_WEIGHT + _TYPE_WEIGHT_STEP*np.maximum(
        np.asarray(weapon_pity) + 1 - _TYPE_WEIGHT_START, 0)
    denominator = np.minimum(character + weapon, _TYPE_WEIGHT_CEILING)
    return np.where(character >= weapon, np.minimum(character, denominator)/denominator,
                    np.maximum(denominator - weapon, 0)/denominator)


@lru_cache(maxsize=1)
def _standard_transitions(pity_p: tuple[float, ...]):
    """两种目标共用状态编码与三种单抽结果，不展开冗余五星水位维度。"""
    limit = _TYPE_CERTAIN_PITY
    maximum_pity = len(pity_p) - 2
    c, w = np.indices((limit + 1, limit + 1))
    valid = np.minimum(c, w) <= maximum_pity
    states = np.stack((c[valid], w[valid]), axis=1)
    space = StateSpace.from_shape([len(states)])
    ids = np.full(valid.shape, -1, dtype=int)
    ids[valid] = np.arange(space.N)
    c, w = states.T
    next_c, next_w = np.minimum(c+1, limit), np.minimum(w+1, limit)
    p5 = np.asarray(pity_p)[np.minimum(c, w) + 1]
    character_probability = standard_5star_type_probability(c, w)
    results = []
    for probability, destination in (
        (1-p5, ids[next_c, next_w]),
        (p5*character_probability, ids[0, next_w]),
        (p5*(1-character_probability), ids[next_c, 0]),
    ):
        active = probability > 0
        builder = ProbabilityMatrixBuilder(space, space)
        builder.add_many(np.flatnonzero(active), destination[active], probability[active])
        results.append(builder.build())
    return space, states, ids, tuple(results)


class StandardGenshin5starModel(GachaModel):
    """常驻池获得任意五星角色或任意五星武器的抽数分布。

    target 为 character 或 weapon；pity_p 是普通五星保底概率表。
    角色/武器等待进度按每抽累计，只在获得同类别五星时归零。
    当前五星水位等于两种进度的较小值，故内部仅保留 (角色进度,武器进度)。
    类别进度达到 179 后，下个五星必为该类别；此后精确合并为同一个状态。
    模型沿用文章参数，仅计算类别，不是某个指定角色或武器。
    """
    def __init__(self, pity_p, target: Literal["character", "weapon"]) -> None:
        table = np.array(pity_p, dtype=float, copy=True)
        if (target not in ("character", "weapon") or table.ndim != 1 or len(table) < 2 or
                len(table)-1 >= _TYPE_WEIGHT_START or not np.all(np.isfinite(table)) or
                np.any(table < 0) or np.any(table > 1) or table[0] != 0 or table[-1] != 1):
            raise ValueError("需要 character/weapon 目标和早于类别软保底的有效五星概率表。")
        self._pity_p = tuple(table)
        self.target = target

    @cached_property
    def _analysis(self) -> HitProcessAnalysis:
        full, states, ids, (low, character, weapon) = _standard_transitions(self._pity_p)
        boundary = StateSpace.from_shape([_TYPE_CERTAIN_PITY + 1])
        if self.target == "character":
            hit_p = np.asarray(character.sum(axis=0)).ravel()
            hit_to = np.minimum(states[:, 1]+1, _TYPE_CERTAIN_PITY)
            reset = ids[0, :]
            miss = low + weapon
        else:
            hit_p = np.asarray(weapon.sum(axis=0)).ravel()
            hit_to = np.minimum(states[:, 0]+1, _TYPE_CERTAIN_PITY)
            reset = ids[:, 0]
            miss = low + character
        hit = sp.csr_matrix((hit_p, (hit_to, np.arange(full.N))), shape=(boundary.N, full.N))
        embed = sp.csr_matrix((np.ones(boundary.N), (reset, np.arange(boundary.N))),
                              shape=(full.N, boundary.N))
        process = HitProcess(full, boundary, miss, hit, embed)
        process.validate()
        # 最迟等待到类别进度 179，再等待一个至多 H 抽的五星周期。
        return HitProcessAnalysis(process, max_steps=_TYPE_CERTAIN_PITY + len(self._pity_p)-1)

    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity: int = 0,
                 character_pity: int | None = None, weapon_pity: int | None = None,
                 *, method: Literal["auto", "direct", "fft"] = "auto") -> FiniteDist | list[FiniteDist]:
        """返回获得 item_num 件目标类别五星的抽数分布。

        item_pity 是当前五星水位；character_pity/weapon_pity 是此前未获得
        五星角色/武器的抽数，较小者必须等于 item_pity。未提供的类别进度
        默认取 item_pity，表示采用两类进度同时归零后累计的初始假设，不是稳态。
        multi_dist=True 返回含零件项的列表；item_num=0 直接返回零花费分布。
        method 控制多件周期卷积；首次用稀疏逐抽递推。保留跨次类别进度。
        """
        validate_steps(item_num)
        if item_num == 0:
            return FiniteDist.delta(0)
        validate_steps(item_pity)
        c = item_pity if character_pity is None else character_pity
        w = item_pity if weapon_pity is None else weapon_pity
        validate_steps(c)
        validate_steps(w)
        if item_pity >= len(self._pity_p)-1 or min(c, w) != item_pity:
            raise ValueError("五星水位须在保底范围内，且等于两类未获得进度的较小值。")
        full, _, ids, _ = _standard_transitions(self._pity_p)
        state = ids[min(c, _TYPE_CERTAIN_PITY), min(w, _TYPE_CERTAIN_PITY)]
        initial = np.zeros(full.N)
        initial[state] = 1
        if item_num == 1:
            result = self._analysis.first_state_dist(initial).marginal_cost(tail_mass=0)
            return [FiniteDist.delta(0), result] if multi_dist else result
        if multi_dist:
            return [FiniteDist.delta(0)] + [joint.marginal_cost(tail_mass=0) for joint in
                self._analysis.iter_state_dists(item_num, initial, method=method)]
        return self._analysis.nth_state_dist(item_num, initial, method=method).marginal_cost(tail_mass=0)
