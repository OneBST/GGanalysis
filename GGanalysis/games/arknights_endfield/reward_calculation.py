import GGanalysis.games.arknights_endfield as AKE
from GGanalysis import FiniteDist, cdf2dist
from typing import Callable, Sequence, Union
from copy import deepcopy
import numpy as np

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

def add_with_last_value_padding(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    将两个1D numpy数组相加。
    若长度不同，则用较短数组的最后一个值补长至相同长度。
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("输入必须是1D numpy数组")
    len_a, len_b = len(a), len(b)
    max_len = max(len_a, len_b)
    if len_a < max_len:
        a = np.pad(a, (0, max_len - len_a), mode='constant', constant_values=a[-1])
    if len_b < max_len:
        b = np.pad(b, (0, max_len - len_b), mode='constant', constant_values=b[-1])
    return a + b

def apply_item_reward_with_distribution(
        raw_cdf_list: list[list[float]],
        reward_dist: np.ndarray,
        reward_pos: int,
    ) -> list[list[float]]:
    n = len(raw_cdf_list) - 1
    reward_cdf_list = [raw_cdf_list[0][:]]  # 0 个道具占位
    for i in range(1, n + 1):
        temp_cdf = np.zeros(1, dtype=float)
        for get_item in range(0, i):
            source_cdf = raw_cdf_list[i-get_item]
            temp_cdf = add_with_last_value_padding(reward_dist[get_item]*source_cdf, temp_cdf)
        temp_cdf = add_with_last_value_padding(np.ones(len(temp_cdf))*sum(reward_dist[i:]), temp_cdf)
        temp_cdf[:reward_pos] = raw_cdf_list[i][:reward_pos]
        reward_cdf_list.append(temp_cdf)
    return reward_cdf_list

if __name__ == '__main__':
    temp = np.zeros(1)
    print(temp[-1])


    test_cdf_list = [np.array([1]), np.array([0, 0, 0.1, 0.2, 0.5, 1]), np.array([0, 0, 0, 0, 0, 0, 1])]
    test_reward_dist = [0.5, 0.25, 0.25]
    ans = apply_item_reward_with_distribution(test_cdf_list, test_reward_dist, 1)
    print(ans)
    exit()
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
