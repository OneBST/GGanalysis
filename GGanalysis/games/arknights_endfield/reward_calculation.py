import GGanalysis.games.arknights_endfield as AKE
from GGanalysis import FiniteDist, cdf2dist
from typing import Callable, Sequence, Union
from copy import deepcopy

RewardRule = Union[
    Callable[[int], int],   # f(j) -> 第 j 次满赠的抽数
    Sequence[int],          # 明确给出每次满赠的位置
]

def reward_positions(
        reward_rule: RewardRule,
        k: int,
    ) -> list[int]:
    # 返回前 k 次满赠的抽数位置（长度 = k）
    if callable(reward_rule):
        pos = [reward_rule(j) for j in range(1, k + 1)]
    else:
        if len(reward_rule) < k:
            raise ValueError(f"reward_rule 列表长度不足：需要 >= {k}")
        pos = list(reward_rule[:k])
    # 基本校验
    if any(p <= 0 for p in pos):
        raise ValueError(f"满赠抽数必须为正整数：{pos}")
    if any(pos[i] >= pos[i + 1] for i in range(len(pos) - 1)):
        raise ValueError(f"满赠抽数必须严格递增：{pos}")
    return pos

def apply_interval_reward(
        raw_cdf_list: list[list[float]],
        reward_rule: RewardRule,
    ) -> list[list[float]]:
    n = len(raw_cdf_list) - 1
    reward_cdf_list = [raw_cdf_list[0][:]]  # 0 个道具占位
    for i in range(1, n + 1):
        temp_cdf = deepcopy(raw_cdf_list[i])
        reward_pos_list = reward_positions(reward_rule, i)
        # j:i 个道具中，有 j 个来自满赠
        for j in range(1, i + 1):
            reward_pos = reward_pos_list[j - 1]
            # 覆盖窗口：到下一次满赠发生前
            end_pos = reward_pos_list[j] if j < i else reward_pos_list[-1]
            src = raw_cdf_list[i - j]
            max_len = min(len(temp_cdf), len(src))
            if reward_pos < max_len:
                right = min(end_pos + 1, max_len)
                temp_cdf[reward_pos:right] = src[reward_pos:right]
            # src 不够长：后面必为 1
            if len(src) < end_pos + 1:
                fill_start = max(len(src), reward_pos)
                fill_end = min(end_pos + 1, len(temp_cdf))
                if fill_start < fill_end:
                    temp_cdf[fill_start:fill_end] = [1.0] * (fill_end - fill_start)
        # 最多只需要到第 i 次满赠发生
        temp_cdf = temp_cdf[: reward_pos_list[-1] + 1]
        reward_cdf_list.append(temp_cdf)
    return reward_cdf_list

if __name__ == '__main__':
    # 处理UP卡池满240额外赠送，先得到没有满赠的模型
    dist_list = AKE.up_6star_first_character(6, multi_dist=True)
    raw_cdf_list = []
    for dist in dist_list:  # 从抽1个开始遍历
        raw_cdf_list.append(dist.cdf)

    # 获得经过240满赠后的cdf
    reward_cdf_list = apply_interval_reward(raw_cdf_list, reward_rule=lambda j: 240 * j)
    # 处理为分布列
    reward_dist_list = []
    for cdf in reward_cdf_list:
        reward_dist_list.append(cdf2dist(cdf))
    for i, dist in zip(range(1, len(reward_dist_list)), reward_dist_list[1:]):
        print(f"获取{i}个UP六星干员所需抽数期望为的{dist.exp}")

    # 计算因为满赠导致达到目标时可能多抽的部分
    def calc_extra_item(dist_list, target_num, interval):
        exp_ans = 0
        for pull_num in range(1, target_num+1):  # 实际抽取个数
            reward_num = target_num - pull_num + 1
            reward_pos = interval(reward_num)
            if reward_pos < len(dist_list[pull_num]):
                exp_ans += dist_list[pull_num][reward_pos]
        return exp_ans

    for i in range(1, len(reward_dist_list)):
        print(f"单抽恰好获取{i}个UP六星干员时获取六星干员的期望个数为{i+calc_extra_item(dist_list, target_num=i, interval=AKE.IntervalAutoReward_UP6star)}")
