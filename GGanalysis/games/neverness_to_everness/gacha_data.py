"""异环棋盘格信息和由其派生的五星概率数据。

棋盘格代码说明
--------------

``I``：初始格。

``L``：学徒宝箱。概率获得 S 级角色或 B 级弧盘：

- 0.2% 概率获得 S 级角色；
- 99.8% 概率获得 B 级弧盘。

``H``：勇者宝箱。以更高概率获得 S 级角色或 B 级弧盘：

- 3% 概率获得 S 级角色；
- 97% 概率获得 B 级弧盘；
- 额外获得 2 个失纬棋子。

失纬棋子的兑换规则：24 个失纬棋子可以兑换一抽，720 个失纬棋子可以兑换
一个常驻 S 级角色。

``W``：弧光盲盒，随机获得一个 A 级弧盘。

``A``：于此同行 A 级角色，100% 获得格子内所展示的 A 级角色。

``G``：于此同行 S 级角色，100% 获得当期格子内所展示的 S 级角色。

以下类型使用下划线后的字段记录获得数量或附加标记：

``SW_n``：失纬棋盒，获得格子内所示数量 ``n`` 的失纬棋子。例如 ``SW_4``
表示获得 4 个失纬棋子，``SW_16`` 表示获得 16 个失纬棋子。

``MD_n``：迷迭棋盒，获得格子内所示数量 ``n`` 的迷迭棋子。70 个迷迭棋子
可以兑换一抽。（每月常驻限定各限5次）

``M_n``：再来一次/多重惊喜，获得格子内所示数量的质实骰子或捏造骰子，
分别对应限定抽和常驻抽。例如 ``M_5`` 表示获得 5 个相应骰子。

``C``：沉眠地，纳库佩达的所在地。需要在 3 次掷骰内追上守护者，成功后可以
获得纳库佩达的私藏，即 30 个失纬棋子。守护者初始会向前逃走 9 格，之后每次
掷骰守护者将向前移动 2 格。

组合标记示例：``W_C`` 表示带沉眠地事件标记的弧光盲盒，``M_1_C`` 表示带
沉眠地事件标记、数量为 1 的骰子奖励格。解析基础格子类型时只读取第一个
下划线之前的部分，因此它们分别归类为 ``W`` 和 ``M``；其余字段继续保留在
原始棋盘信息中，供以后实现附加机制时使用。

角色池特有格子：

``HD1``：今日穿搭，获得角色皮肤。

``HD2``：改装时刻，获得载具涂装。

``HD3``：风向标，必定获得滑翔翼皮肤。

当已经连续 70 抽没有获得 S 级角色时，会从普通棋盘切换到变格棋盘。对应数据
分别记录在 ``STANDARD_MAP_INFO`` / ``CHARACTER_MAP_INFO`` 与
``STANDARD_MAP_PITY_INFO`` / ``CHARACTER_MAP_PITY_INFO`` 中。
"""

from typing import Sequence

HARD_PITY = 90
END_POS = 73
PITY_START = 70

# 格子类型对应的五星角色概率；未列出的类型概率为 0。
TILE_5STAR_PROBABILITY = {
    "L": 0.002,
    "H": 0.03,
    "G": 1.0,
}


# 常驻池棋盘信息
STANDARD_MAP_INFO = [
    "I",
    "A", "L", "SW_4", "L", "L", "H",
    "L", "L", "MD_30", "A", "L", "H",
    "L", "L", "MD_50", "L", "A", "H",
    "L", "L", "MD_30", "W_C", "L", "H",
    "L", "L", "MD_50", "L", "A", "H",
    "L", "L", "MD_30", "L", "L", "H",
    "A", "L", "MD_30", "L", "MD_50", "H",
    "L", "A", "MD_30", "L", "L", "H",
    "M_1_C", "L", "W", "MD_30", "L", "H",
    "G", "SW_16", "H", "L", "H", "L", "H", "L", "H",
    "H", "SW_16", "M_5", "H", "H", "H", "H", "H", "H",
]

STANDARD_MAP_PITY_INFO = [
    "I",
    "A", "L", "SW_4", "L", "L", "G",
    "L", "L", "MD_30", "A", "L", "G",
    "L", "L", "MD_50", "L", "A", "G",
    "L", "L", "MD_30", "W_C", "L", "G",
    "L", "L", "MD_50", "L", "A", "G",
    "L", "L", "MD_30", "L", "L", "G",
    "A", "L", "MD_30", "L", "MD_50", "G",
    "L", "A", "MD_30", "L", "L", "G",
    "M_1_C", "L", "W", "MD_30", "L", "G",
    "G", "SW_16", "G", "L", "G", "L", "G", "L", "G",
    "G", "SW_16", "M_5", "H", "G", "H", "G", "H", "G",
]

# 角色池棋盘信息
CHARACTER_MAP_INFO = [
    "I",
    "A", "L", "HD3", "L", "L", "H",
    "L", "L", "MD_30", "A", "L", "H",
    "L", "L", "MD_50", "L", "A", "H",
    "L", "L", "MD_30", "W_C", "L", "H",
    "L", "L", "MD_50", "L", "A", "H",
    "L", "L", "MD_30", "L", "L", "H",
    "A", "L", "MD_30", "L", "MD_50", "H",
    "L", "A", "MD_30", "L", "L", "H",
    "M_1_C", "L", "W", "MD_30", "L", "H",
    "G", "HD2", "H", "L", "H", "L", "H", "L", "H",
    "H", "HD1", "M_5", "H", "H", "H", "H", "H", "H",
]

CHARACTER_MAP_PITY_INFO = [
    "I",
    "A", "L", "HD3", "L", "L", "G",
    "L", "L", "MD_30", "A", "L", "G",
    "L", "L", "MD_50", "L", "A", "G",
    "L", "L", "MD_30", "W_C", "L", "G",
    "L", "L", "MD_50", "L", "A", "G",
    "L", "L", "MD_30", "L", "L", "G",
    "A", "L", "MD_30", "L", "MD_50", "G",
    "L", "A", "MD_30", "L", "L", "G",
    "M_1_C", "L", "W", "MD_30", "L", "G",
    "G", "HD2", "G", "L", "G", "L", "G", "L", "G",
    "G", "HD1", "M_5", "H", "G", "H", "G", "H", "G",
]


def parse_tile_type(tile_info: str) -> str:
    """从 ``SW_4``、``M_1_C`` 等格子信息中解析基础类型。"""
    return tile_info.split("_", 1)[0]


def parse_5star_probability_map(map_info: Sequence[str]) -> tuple[float, ...]:
    """根据 73 格棋盘信息生成每个位置获得S角色的概率。"""
    if len(map_info) != END_POS:
        raise ValueError(f"map_info must contain {END_POS} positions, got {len(map_info)}.")
    return tuple(
        TILE_5STAR_PROBABILITY.get(parse_tile_type(tile), 0.0)
        for tile in map_info
    )


# 各卡池在普通棋盘和变格棋盘上的五星概率图。
# 池类型和棋盘阶段都写入名称，避免把“普通阶段”与“常驻池”混为一谈。
STANDARD_POOL_NORMAL_MAP_5 = parse_5star_probability_map(STANDARD_MAP_INFO)
STANDARD_POOL_PITY_MAP_5 = parse_5star_probability_map(STANDARD_MAP_PITY_INFO)
CHARACTER_POOL_NORMAL_MAP_5 = parse_5star_probability_map(CHARACTER_MAP_INFO)
CHARACTER_POOL_PITY_MAP_5 = parse_5star_probability_map(CHARACTER_MAP_PITY_INFO)


__all__ = [
    "HARD_PITY",
    "END_POS",
    "PITY_START",
    "TILE_5STAR_PROBABILITY",
    "STANDARD_MAP_INFO",
    "STANDARD_MAP_PITY_INFO",
    "CHARACTER_MAP_INFO",
    "CHARACTER_MAP_PITY_INFO",
    "STANDARD_POOL_NORMAL_MAP_5",
    "STANDARD_POOL_PITY_MAP_5",
    "CHARACTER_POOL_NORMAL_MAP_5",
    "CHARACTER_POOL_PITY_MAP_5",
    "parse_tile_type",
    "parse_5star_probability_map",
]
