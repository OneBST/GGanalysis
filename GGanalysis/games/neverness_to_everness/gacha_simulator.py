"""异环角色池棋盘的蒙特卡洛模拟器。

本模块直接逐抽模拟掷骰、棋盘移动、格子奖励和 pity，不使用
``gacha_model.py`` 构造的 Markov 转移矩阵，可用于独立交叉验证解析结果。
"""

from dataclasses import dataclass
import random

from tqdm import tqdm

from GGanalysis.games.neverness_to_everness.gacha_data import (
    CHARACTER_MAP_INFO,
    CHARACTER_MAP_PITY_INFO,
    HARD_PITY,
    PITY_START,
    TILE_5STAR_PROBABILITY,
    parse_tile_type,
)


__all__ = [
    "NTESimulationResult",
    "simulate_nte_statistics",
    "print_simulation_comparison",
]


def _move_position(position: int, dice: int) -> int:
    """按照棋盘岔路规则移动一颗六面骰的点数。"""
    if position == 16:
        return 54 + dice
    if position == 43:
        return 63 + dice
    if 55 <= position <= 63:
        return position + dice if position + dice <= 63 else position + dice - 64 + 18
    if 64 <= position <= 72:
        return position + dice if position + dice <= 72 else position + dice - 73 + 45
    return (position + dice - 1) % 54 + 1


def _split_tile_events(tile_info: str) -> tuple[str, bool]:
    """把格子拆为互斥奖励代码和是否同时触发 ``C`` 事件。"""
    fields = tile_info.split("_")
    has_c_event = "C" in fields[1:]
    reward = "_".join(field for field in fields if field != "C")
    return reward, has_c_event


@dataclass(frozen=True)
class NTESimulationResult:
    """一次长期模拟得到的计数及其经验统计量。"""

    pulls: int
    burn_in: int
    seed: int | None
    five_star_count: int
    roll_count: int
    reward_counts: dict[str, int]
    c_trigger_count: int
    catch_count: int
    complete_cycle_count: int
    complete_cycle_pulls: int

    @property
    def five_star_probability(self) -> float:
        return self.five_star_count / self.pulls

    @property
    def roll_probability(self) -> float:
        return self.roll_count / self.pulls

    @property
    def mean_cycle_pulls(self) -> float:
        if self.complete_cycle_count == 0:
            return float("nan")
        return self.complete_cycle_pulls / self.complete_cycle_count

    @property
    def c_trigger_probability(self) -> float:
        return self.c_trigger_count / self.pulls

    @property
    def catch_probability_per_trigger(self) -> float:
        if self.c_trigger_count == 0:
            return float("nan")
        return self.catch_count / self.c_trigger_count

    @property
    def catch_probability_per_pull(self) -> float:
        return self.catch_count / self.pulls

    @property
    def tile_entry_probabilities(self) -> dict[str, float]:
        """返回每抽进入各奖励格及附加 ``C`` 事件的经验概率。"""
        probability = {
            reward: count / self.pulls
            for reward, count in sorted(self.reward_counts.items())
        }
        probability["C"] = self.c_trigger_probability
        return dict(sorted(probability.items()))


def simulate_nte_statistics(
    pulls: int = 500_000,
    *,
    burn_in: int = 10_000,
    seed: int | None = 20260713,
    show_progress: bool = True,
) -> NTESimulationResult:
    """逐抽模拟异环角色池，返回长期统计量。

    ``burn_in`` 次预热不计入结果，用于消除从 ``pity=0, position=0`` 开始造成的
    短期初始偏差。第 90 抽硬保底原地获得 S，不掷骰也不重复获得当前位置奖励。
    """
    if not isinstance(pulls, int) or pulls <= 0:
        raise ValueError("pulls must be a positive integer.")
    if not isinstance(burn_in, int) or burn_in < 0:
        raise ValueError("burn_in must be a non-negative integer.")

    # 轨迹的下一步依赖当前状态，无法一次向量化生成；标准库标量随机数比反复调用
    # NumPy 的单元素接口更适合这里。
    rng = random.Random(seed)
    pity = 0
    position = 0
    five_star_count = 0
    roll_count = 0
    reward_counts: dict[str, int] = {}
    c_trigger_count = 0
    catch_count = 0
    complete_cycle_count = 0
    complete_cycle_pulls = 0
    last_recorded_hit: int | None = None

    steps = range(burn_in + pulls)
    if show_progress:
        steps = tqdm(steps, desc="模拟异环抽取", unit="抽")

    for step in steps:
        recorded = step >= burn_in
        obtained_five_star = pity == HARD_PITY - 1

        if not obtained_five_star:
            dice = rng.randint(1, 6)
            position = _move_position(position, dice)
            map_info = (
                CHARACTER_MAP_PITY_INFO
                if pity >= PITY_START
                else CHARACTER_MAP_INFO
            )
            tile_info = map_info[position]
            reward, has_c_event = _split_tile_events(tile_info)

            if recorded:
                roll_count += 1
                reward_counts[reward] = reward_counts.get(reward, 0) + 1

            if has_c_event:
                dice_1 = rng.randint(1, 6)
                dice_2 = rng.randint(1, 6)
                dice_3 = rng.randint(1, 6)
                caught = dice_1 + dice_2 >= 11 or dice_1 + dice_2 + dice_3 >= 13
                if recorded:
                    c_trigger_count += 1
                    catch_count += int(caught)

            obtain_probability = TILE_5STAR_PROBABILITY.get(parse_tile_type(tile_info), 0.0)
            obtained_five_star = rng.random() < obtain_probability

        if obtained_five_star:
            pity = 0
            if recorded:
                five_star_count += 1
                if last_recorded_hit is not None:
                    complete_cycle_count += 1
                    complete_cycle_pulls += step - last_recorded_hit
                last_recorded_hit = step
        else:
            pity += 1

    return NTESimulationResult(
        pulls=pulls,
        burn_in=burn_in,
        seed=seed,
        five_star_count=five_star_count,
        roll_count=roll_count,
        reward_counts=reward_counts,
        c_trigger_count=c_trigger_count,
        catch_count=catch_count,
        complete_cycle_count=complete_cycle_count,
        complete_cycle_pulls=complete_cycle_pulls,
    )


def print_simulation_comparison(
    pulls: int = 100_000_000,
    *,
    burn_in: int = 10_000,
    seed: int | None = 20260713,
    show_progress: bool = True,
) -> NTESimulationResult:
    """运行模拟并打印解析值、模拟值和两者差值。"""
    from GGanalysis.games.neverness_to_everness.gacha_model import (
        character_stationary,
        up_5star_character,
    )

    result = simulate_nte_statistics(
        pulls,
        burn_in=burn_in,
        seed=seed,
        show_progress=show_progress,
    )
    exact_tile = character_stationary.tile_entry_probabilities
    simulated_tile = result.tile_entry_probabilities
    exact_nacupeda = character_stationary.nacupeda_statistics

    print(f"=== 蒙特卡洛交叉验证（{pulls:,} 抽，预热 {burn_in:,} 抽，seed={seed}）===")
    print("统计量                       解析值          模拟值          差值")
    summary = [
        ("长期 S 概率", up_5star_character.stationary_probability, result.five_star_probability),
        ("相邻 S 平均抽数", character_stationary.steady_dist.exp, result.mean_cycle_pulls),
        ("实际掷骰概率", float(character_stationary.position_entry_probabilities.sum()), result.roll_probability),
        ("C 事件每抽概率", exact_nacupeda.trigger_probability_per_pull, result.c_trigger_probability),
        ("三骰追上条件概率", exact_nacupeda.catch_probability_per_trigger, result.catch_probability_per_trigger),
        ("每抽触发并追上概率", exact_nacupeda.catch_probability_per_pull, result.catch_probability_per_pull),
    ]
    for name, exact, simulated in summary:
        print(f"{name:<25} {exact:>12.8f} {simulated:>12.8f} {simulated - exact:>+12.8f}")

    print("\n格子       解析每抽概率    模拟每抽概率          差值")
    for tile in sorted(exact_tile.keys() | simulated_tile.keys()):
        exact = exact_tile.get(tile, 0.0)
        simulated = simulated_tile.get(tile, 0.0)
        print(f"{tile:<8} {exact:>14.9%} {simulated:>14.9%} {simulated - exact:>+13.9%}")

    return result


if __name__ == "__main__":
    print_simulation_comparison()
