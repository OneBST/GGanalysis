"""
异环 Neverness to Everness (NTE) 抽卡模型（测试服版）
基于 GGanalysis.markov 模块重新实现。

角色池核心机制：
- 投骰子走棋盘：每次投骰 1-6 概率均等（各 1/6），移动对应步数
- 棋盘共 73 个位置（0-72），位置 0 为起点，每个位置有不同的 S 级获取概率
- 第 1-70 抽使用标准概率图，第 71-89 抽触发全局概率提升图
- 第 90 抽硬保底：不移动，直接获得 S 级角色
- 位置 16 / 43 分别进入特殊区域 1（格子 55-63）/ 特殊区域 2（格子 64-72）
- 离开特殊区域 1 / 2 后分别到达位置 18 / 45
- 没有大小保底设计，S 级即为限定角色
- 公示综合概率 1.88%，期望约 53.07 抽

实现方法：
- 使用 StateSpace 定义二维状态空间 [pity(0-89), position(0-72)]
- 使用 TransitionBuilder 构建 CSR 稀疏转移矩阵（6,570 × 6,570）
- 使用 stationary_solve 求解平稳分布
- 使用 first_hitting_time 计算条件分布 PMF
- 通过平稳分布法与条件分布法交叉验证

参考资料：
- https://www.bilibili.com/opus/1083112283021770769
- https://www.bilibili.com/opus/1085754117440143393
"""

from functools import lru_cache
import numpy as np
from typing import Union, List
from GGanalysis.basic_models import GachaModel
from GGanalysis.distribution_1d import FiniteDist, calc_expectation
from GGanalysis.markov.state_space import StateSpace
from GGanalysis.markov.transition import MarkovTransition
from GGanalysis.markov.builder import TransitionBuilder
from GGanalysis.markov.analysis import stationary_solve, first_hitting_time

__all__ = [
    'STANDARD_MAP_S',
    'PITY_MAP_S',
    'build_nte_transition',
    'get_steady_single_dist',
    'get_c_dist',
    'character_s',
    'NTECharacterSModel',
]

# 棋盘格获得S概率定义
N = 0.0
L = 0.002
H = 0.03
G = 1.0
# 棋盘转移定义
'''
踩到第16格/43格进入特殊区域1/2
从特殊区域1/2离开后进入第18格/45格
思路为构造爬山模型，90抽保底，则构造 0-89 层结构
每层表示保底进度，每一层要么是没有中S级向本下一层内下一个地块前进，
要么是中S级了回到第0层的下一个地块
第89层不会进行移动，直接回到第0层的对应位置
第0格用于作为起始点使用
第71抽开始触发保底棋盘，概率提升
'''
STANDARD_MAP_S = np.array([
    0,                 # 0  初始格
    N, L, N, L, L, H,  # 1-6
    L, L, N, N, L, H,  # 7-12
    L, L, N, L, N, H,  # 13-18（16 为特殊区域 1 入口）
    L, L, N, N, L, H,  # 19-24
    L, L, N, L, N, H,  # 25-30
    L, L, N, L, L, H,  # 31-36
    N, L, N, L, N, H,  # 37-42
    L, N, N, L, L, H,  # 43-48（43 为特殊区域 2 入口）
    N, L, N, N, L, H,  # 49-54
    G, L, H, L, H, L, H, L, H,  # 55-63 特殊区域 1 内部
    H, H, N, H, H, H, H, H, H,  # 64-72 特殊区域 2 内部
], dtype=np.float64)

PITY_MAP_S = np.array([
    0,                 # 0  初始格
    N, H, N, L, L, G,  # 1-6
    L, L, N, N, L, G,  # 7-12
    L, H, N, L, N, G,  # 13-18（16 为特殊区域 1 入口）
    L, L, N, N, L, G,  # 19-24
    L, H, N, L, N, G,  # 25-30
    L, L, N, L, L, G,  # 31-36
    N, H, N, L, N, G,  # 37-42
    L, N, N, L, L, G,  # 43-48（43 为特殊区域 2 入口）
    N, H, N, N, L, G,  # 49-54
    G, L, G, L, G, L, G, L, G,  # 55-63 特殊区域 1 内部
    G, H, N, H, G, H, G, H, G,  # 64-72 特殊区域 2 内部
], dtype=np.float64)

# 状态空间维度常量
HARD_PITY = 90   # 保底计数 0-89
END_POS = 73    # 棋盘位置 0-72


# 定义棋盘转移
def _get_next_positions(pos: int) -> List[tuple]:
    """
    根据当前位置，计算掷骰子 1-6 后可能到达的位置及对应概率
    """
    ans = []
    if pos == 16:
        # 踩到入口格，进入特殊区域 1（格子 55-63）
        for i in range(1, 7):
            ans.append((54 + i, 1/6))
        return ans
    if pos == 43:
        # 踩到入口格，进入特殊区域 2（格子 64-72）
        for i in range(1, 7):
            ans.append((63 + i, 1/6))
        return ans
    if 55 <= pos <= 63:
        # 在特殊区域 1 内部移动，离开后到达位置 18
        for i in range(1, 7):
            next_pos = pos + i
            if next_pos > 63:
                next_pos = next_pos - 64 + 18
            ans.append((next_pos, 1/6))
        return ans
    if 64 <= pos <= 72:
        # 在特殊区域 2 内部移动，离开后到达位置 45
        for i in range(1, 7):
            next_pos = pos + i
            if next_pos > 72:
                next_pos = next_pos - 73 + 45
            ans.append((next_pos, 1/6))
        return ans
    # 常规棋盘位置（1-54），循环移动
    for i in range(1, 7):
        ans.append(((pos + i - 1) % 54 + 1, 1/6))
    return ans


# 转移矩阵构建
@lru_cache
def build_nte_transition() -> MarkovTransition:
    """
    构建异环角色池获得 S 级角色的完整转移矩阵。

    状态空间定义为 [pity, position]：
    - pity (0-89)：当前连续未出 S 级的抽数
    - position (0-72)：当前在棋盘上的位置

    共 90×73 = 6570 个状态，使用 CSR 稀疏格式存储。

    转移规则：
    - pity < 89：掷骰子移动后，按所踩格子的概率决定是否获得 S 级
      - 获得 S 级 → pity 重置为 0，位置更新
      - 未获得   → pity + 1，位置更新
    - pity = 89：必定获得 S 级，pity 重置为 0，位置不变（不移动直接获得）
    """
    nte_statespace = StateSpace([HARD_PITY - 1, END_POS - 1])
    builder = TransitionBuilder(nte_statespace, backend="sparse")

    for pity in range(HARD_PITY):
        for pos in range(END_POS):
            from_state = np.array([pity, pos], dtype=np.int64)
            if pity == 89:
                # 第 90 抽硬保底：不移动，直接获得，保底清零
                builder.add_state(from_state, np.array([0, pos], dtype=np.int64), 1.0)
                continue
            # 70 抽后触发保底棋盘
            map_p = PITY_MAP_S if pity >= 70 else STANDARD_MAP_S
            for next_pos, dice_prob in _get_next_positions(pos):
                s_prob = float(map_p[next_pos])
                # 获得 S 级：保底清零
                if s_prob > 0:
                    builder.add_state(
                        from_state,
                        np.array([0, next_pos], dtype=np.int64),
                        s_prob * dice_prob,
                    )
                # 未获得 S 级：保底 +1
                if s_prob < 1.0:
                    builder.add_state(
                        from_state,
                        np.array([pity + 1, next_pos], dtype=np.int64),
                        (1.0 - s_prob) * dice_prob,
                    )
    return builder.build(check=True)

@lru_cache
def get_steady_single_dist() -> FiniteDist:
    """
    获取平稳态下获取 1 个 S 级角色的抽数分布。

    从平稳分布 π 中提取 pity=0（刚获得 S）的部分作为初始分布，
    计算下一次获得 S 的首达分布，即相邻两次 S 级的间隔抽数分布。
    """
    tm = build_nte_transition()
    pi = stationary_solve(tm)

    # 从平稳分布提取 pity=0 部分，归一化作为初始分布
    ss = tm.space
    pity0_ids = ss.select_ids([0, None])
    p0_post_hit = ss.zeros()
    p0_post_hit[pity0_ids] = pi[pity0_ids]
    p0_post_hit /= p0_post_hit.sum()

    f, _ = first_hitting_time(
        tm, p0_post_hit, [0, None], HARD_PITY, include_t0=False
    )
    return FiniteDist(f)

@lru_cache
def get_c_dist(init_pos: int = 0, init_pity: int = 0) -> FiniteDist:
    """
    获取从指定初始状态出发时，获取 1 个 S 级角色的条件分布。
    """
    tm = build_nte_transition()
    init_dist = tm.space.delta([init_pity, init_pos])
    dist_pmf, _ = first_hitting_time(
        tm, init_dist, [0, None], HARD_PITY, include_t0=False
    )
    return FiniteDist(dist_pmf)

class NTECharacterSModel(GachaModel):
    """
    异环角色池 S 级角色抽卡模型。

    基于马尔科夫链精确计算，纳入：
    - 棋盘移动规则（含特殊区域进出）
    - 标准 / 保底概率图切换
    - 90 抽硬保底
    - 无大小保底（S 级即为限定）

    多道具计算采用位置感知的迭代卷积：
    每次获得 S 级后的落点位置会影响下一次的分布，
    因此逐道具迭代，在每一步追踪位置分布并准确传播。

    使用方法
    --------
    >>> from GGanalysis.games.neverness_to_everness import character_s
    >>> dist = character_s(1)                     # 单 S 级
    >>> dists = character_s(6, multi_dist=True)   # [1..6] 个的分布列表
    >>> dist = character_s(1, start_pos=16, item_pity=30)  # 指定初始状态
    """

    def __init__(self) -> None:
        self.tm = build_nte_transition()
        self.ss = self.tm.space
        self.steady_dist = get_steady_single_dist()

        # 预计算：从每个 pity=0 位置出发的首达分布，并堆叠为矩阵加速卷积
        # pull_matrix[τ, k] = P(τ 抽获得 S | 从位置 k 出发)
        # joint_tensor[τ, p, k] = P(τ 抽获得 S, 落在 p | 从位置 k 出发)
        pull_cols, joint_cols = [], []
        for pos in range(END_POS):
            init_dist = self.ss.delta([0, pos])
            pull_pmf, _, p_after = first_hitting_time(
                self.tm, init_dist, [0, None], HARD_PITY,
                include_t0=False, return_pos=True,
            )
            pull_cols.append(pull_pmf)
            joint_cols.append(pull_pmf[:, None] * p_after)  # 联合: (91, 73)
        self._pull_matrix = np.column_stack(pull_cols)       # (91, 73)
        self._joint_tensor = np.stack(joint_cols, axis=2)     # (91, 73, 73)  [τ, p, k]
        # 预展平 joint_tensor 供批量矩阵乘法: (91*73, 73)
        self._joint_flat = self._joint_tensor.reshape(-1, END_POS)

    def __call__(
        self,
        item_num: int = 1,
        multi_dist: bool = False,
        start_pos: int = 0,
        item_pity: int = 0,
    ) -> Union[FiniteDist, list]:
        """
        计算获取 item_num 个 S 级角色的抽数分布。

        多道具采用位置感知卷积：追踪 P(累计抽数=t, 当前位置=p) 的联合分布，
        每轮卷积时按上轮结束位置选择对应的预计算首达分布进行叠加。

        Parameters
        ----------
        item_num : int
            目标获取个数。
        multi_dist : bool
            为 True 时返回列表，包含获取 [1..item_num] 个的分布。
        start_pos : int
            初始棋盘位置 (0-72)。
        item_pity : int
            初始保底计数 (0-89)，即已垫抽数。

        Returns
        -------
        FiniteDist 或 list[FiniteDist]
        """
        if item_num == 0:
            return FiniteDist([1])

        # 第一个道具：从初始状态出发
        init_dist = self.ss.delta([item_pity, start_pos])
        pull_pmf, _, pos_cond = first_hitting_time(
            self.tm, init_dist, [0, None], HARD_PITY,
            include_t0=False, return_pos=True,
        )
        pull_dist = pull_pmf    # 边际抽数 PMF
        pos_dist = pos_cond     # 条件位置分布 (91, 73)

        if item_num == 1:
            return FiniteDist(pull_dist)  # 单道具，概率和≈1，不需要归一化

        if multi_dist:
            ans_list = [FiniteDist([1]), FiniteDist(pull_dist)]

        for _ in range(2, item_num + 1):
            max_len = len(pull_dist) + HARD_PITY
            next_pull = np.zeros(max_len, dtype=np.float64)
            next_pos = np.zeros((max_len, END_POS), dtype=np.float64)

            for i in range(1, len(pull_dist)):
                prob_i = pull_dist[i]  # P(前几个道具共花 i 抽)
                if prob_i == 0:
                    continue
                prob_k = prob_i * pos_dist[i, :]  # (73,) P(i 抽, 位置=k)

                # 边际卷积：pull_matrix 的 k 列是从位置 k 出发的抽数 PMF
                next_pull[i:i + HARD_PITY + 1] += self._pull_matrix @ prob_k

                # 联合累积：一次性算出对所有 (τ 抽, 位置 p) 的贡献
                contrib = (self._joint_flat @ prob_k).reshape(HARD_PITY + 1, END_POS)
                next_pos[i:i + HARD_PITY + 1, :] += contrib

            pull_dist = next_pull
            pos_dist = next_pos.copy()
            for p in range(1, len(pull_dist)):
                if pull_dist[p] > 0:
                    pos_dist[p, :] /= pull_dist[p]  # 联合 → 条件
            if multi_dist:
                ans_list.append(FiniteDist(pull_dist / pull_dist.sum()))

        if multi_dist:
            return ans_list

        return FiniteDist(pull_dist / pull_dist.sum())

    def conditional_dist(self, init_pos: int = 0, init_pity: int = 0) -> FiniteDist:
        """获取从指定初始状态出发时，获取 1 个 S 级角色的条件分布。"""
        return get_c_dist(init_pos, init_pity)

    @property
    def stationary_probability(self) -> float:
        """S 级综合概率（平稳态下单抽获得 S 级的概率）。"""
        pi = stationary_solve(self.tm)
        pity0_ids = self.ss.select_ids([0, None])
        return float(pi[pity0_ids].sum())


# ==================== 预构建模型实例 ====================

character_s = NTECharacterSModel()


# ==================== 测试 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("异环二测角色池 S 级角色模型测试")
    print("基于 GGanalysis.markov 模块")
    print("=" * 60)

    tm = build_nte_transition()
    print(f"\n转移矩阵规模: {tm.N:,} × {tm.N:,}")
    print(f"矩阵后端:      {tm.backend}")

    # 平稳分布分析
    pi = stationary_solve(tm)
    pity0_ids = tm.space.select_ids([0, None])
    prob = float(pi[pity0_ids].sum())
    print(f"\nS 级综合概率: {prob * 100:.5f}%（公示 1.8800%）")

    # 单 S 级稳态分布
    f_dist = get_steady_single_dist()
    print(f"\n获取单个 S 级角色（稳态）:")
    print(f"  期望抽数:     {f_dist.exp:.2f}")
    print(f"  标准差:       {f_dist.var ** 0.5:.2f}")
    print(f"  变异系数:     {f_dist.var ** 0.5 / f_dist.exp:.4f}")
    print(f"  71 抽获得概率: {f_dist[71] * 100:.4f}%")
    print(f"  90 抽保底概率: {f_dist[90] * 100:.4f}%")

    # 多 S 级期望
    print(f"\n获取多个 S 级角色期望抽数:")
    dists = character_s(6, multi_dist=True)
    for i, d in enumerate(dists[1:], 1):
        print(f"  {i} 个: {d.exp:.2f} 抽 (平均每个 {d.exp / i:.2f} 抽)")

    # 不同初始状态的条件分布
    print(f"\n不同初始状态的条件分布期望:")
    for init_pos in [0, 16, 43]:
        for init_pity in [0, 30, 70]:
            c = get_c_dist(init_pos, init_pity)
            print(f"  pos={init_pos:2d}, pity={init_pity:2d}: 期望 {c.exp:.2f} 抽")

    # 与原神对比
    print(f"\n--- 跨游戏对比（变异系数） ---")
    print(f"  异环 S 级: {f_dist.var ** 0.5 / f_dist.exp:.4f}")

    try:
        from GGanalysis.games.genshin_impact import up_5star_character
        gi_dist = up_5star_character(1)
        print(f"  原神 UP 五星: {gi_dist.var ** 0.5 / gi_dist.exp:.4f}")
    except ImportError:
        print(f"  原神: (未导入)")

    # ============================
    # 交叉验证：first_hitting_time 方法 vs stationary 方法
    # 从平稳分布中提取 pity=0（即"刚刚获得 S 级"）的分布作为初始分布，
    # 使用 first_hitting_time 计算条件分布，结果应与 f_dist 一致
    # ============================
    print(f"\n--- 交叉验证 ---")
    pity0_ids = tm.space.select_ids([0, None])
    p0_post_hit = tm.space.zeros()
    p0_post_hit[pity0_ids] = pi[pity0_ids]
    p0_post_hit /= p0_post_hit.sum()
    f_fht, surv = first_hitting_time(tm, p0_post_hit, [0, None], HARD_PITY, include_t0=False)

    fht_dist = FiniteDist(f_fht)
    max_diff = np.max(np.abs(f_dist.dist[:HARD_PITY + 1] - f_fht[:HARD_PITY + 1]))
    print(f"  stationary 方法期望:      {f_dist.exp:.4f}")
    print(f"  first_hitting_time 期望:  {fht_dist.exp:.4f}")
    print(f"  期望差异:                 {abs(f_dist.exp - fht_dist.exp):.4f}")
    print(f"  PMF 最大绝对差异:         {max_diff:.2e}")
    print(f"  {'[PASS]' if max_diff < 1e-8 else '[FAIL]'} 两种方法交叉验证")
