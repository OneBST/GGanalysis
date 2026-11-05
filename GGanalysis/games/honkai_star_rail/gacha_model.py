'''
    本模块五星及四星概率上升模型可信度很高，其余部分可能存在没有完全覆盖的机制
'''
import numpy as np
from typing import Literal
from GGanalysis.basic_models import DualPityBernoulliModel, DualPityModel, PityModel
from GGanalysis.distribution_1d import FiniteDist, p2dist

__all__ = [
    'PITY_5STAR',
    'PITY_4STAR',
    'PITY_W5STAR',
    'PITY_W4STAR',
    'common_5star',
    'common_4star',
    'up_5star_character',
    'up_4star_character',
    'up_4star_specific_character',
    'common_5star_weapon',
    'common_4star_weapon',
    'up_5star_weapon',
    'up_4star_weapon',
    'up_4star_specific_weapon',
    'PHANTASIA_SHOWSTAGE_PITY',
    'PHANTASIA_SHOWSTAGE_INVITATION_COST',
    'PHANTASIA_SHOWSTAGE_JADE_PER_INVITATION',
    'phantasia_showstage',
]

# 星穹铁道普通5星保底概率表
PITY_5STAR = np.zeros(91)
PITY_5STAR[1:74] = 0.006
PITY_5STAR[74:90] = np.arange(1, 17) * 0.06 + 0.006
PITY_5STAR[90] = 1
# 星穹铁道普通4星保底概率表
PITY_4STAR = np.zeros(11)
PITY_4STAR[1:9] = 0.051
PITY_4STAR[9] = 0.051 + 0.51
PITY_4STAR[10] = 1
# 星穹铁道武器池5星保底概率表
PITY_W5STAR = np.zeros(81)
PITY_W5STAR[1:66] = 0.008
PITY_W5STAR[66:80] = np.arange(1, 15) * 0.07 + 0.008
PITY_W5STAR[80] = 1
# 星穹铁道武器池4星保底概率表
PITY_W4STAR = np.array([0,0.066,0.066,0.066,0.066,0.066,0.066,0.066,0.466,0.866,1])

# 定义获取星级物品的模型
common_5star = PityModel(PITY_5STAR)
common_4star = PityModel(PITY_4STAR)
# 定义星穹铁道角色池模型
up_5star_character = DualPityModel(PITY_5STAR, [0, 0.5 + 0.5/8, 1])
up_4star_character = DualPityModel(PITY_4STAR, [0, 0.5, 1])
up_4star_specific_character = DualPityBernoulliModel(PITY_4STAR, [0, 0.5, 1], 1/3)
# 定义星穹铁道武器池模型
common_5star_weapon = PityModel(PITY_W5STAR)
common_4star_weapon = PityModel(PITY_W4STAR)
up_5star_weapon = DualPityModel(PITY_W5STAR, [0, 0.75 + 0.25/8, 1])
up_4star_weapon = DualPityModel(PITY_W4STAR, [0, 0.75, 1])
up_4star_specific_weapon = DualPityBernoulliModel(PITY_W4STAR, [0, 0.75, 1], 1/3)

# 星穹铁道时装池
# 时装抽取每次获得一项并移出奖池，不重复。抽中时装时同时获得池内全部剩余奖品
# 首抽免费，之后逐次消耗 3/5/8/12/15/20/24 张时装邀请函，每花费 100 星琼兑换 1 张
# 最多消耗 87 张（8700 星琼）。消耗的邀请函计入「万象和鸣」，用于获取「奇缘星魂」
PHANTASIA_SHOWSTAGE_PITY = (0, 0.0005, 0.0025, 0.01, 0.015, 0.02, 0.05, 0.25, 1)
# 索引 k 为第 k 抽的单次消耗，索引 0 为占位，索引 1 为免费首抽。
PHANTASIA_SHOWSTAGE_INVITATION_COST = (0, 0, 3, 5, 8, 12, 15, 20, 24)
PHANTASIA_SHOWSTAGE_JADE_PER_INVITATION = 100


def phantasia_showstage(
    unit: Literal['pull', 'invitation', 'stellar_jade'] = 'pull',
) -> FiniteDist:
    '''返回幻乐秀场首次获得「群星乐透」时装的累计花费分布。

    Parameters
    ----------
    unit : {'pull', 'invitation', 'stellar_jade'}, optional
        花费单位：抽数（默认，包含免费首抽）、时装邀请函或星琼。
        星琼按所有消耗的邀请函均以 100 星琼兑换计算。

    Returns
    -------
    FiniteDist
        ``dist[t]`` 为恰好花费 t 单位获得时装的概率，``exp`` 为期望。
        邀请函和星琼分布在索引 0 保留免费首抽命中的概率。

    Notes
    -----
    初始状态为尚未参与的完整奖池，抽中时装即停止。采用公示的逐抽条件
    概率进行有限递推，无截断或模拟。不支持重复获取时装、跨期继承或
    非时装奖品首次获取分析，也不抵扣「万象和鸣」奖励的价值。
    '''
    if unit not in ('pull', 'invitation', 'stellar_jade'):
        raise ValueError("unit 必须为 'pull'、'invitation' 或 'stellar_jade'")
    pull_dist = p2dist(PHANTASIA_SHOWSTAGE_PITY)
    if unit == 'pull':
        return pull_dist
    costs = np.cumsum(PHANTASIA_SHOWSTAGE_INVITATION_COST)
    if unit == 'stellar_jade':
        costs *= PHANTASIA_SHOWSTAGE_JADE_PER_INVITATION
    probabilities = np.zeros(int(costs[-1]) + 1)
    # 合并映射至同一花费的概率，包含零花费的免费首抽。
    np.add.at(probabilities, costs, pull_dist.dist)
    return FiniteDist(probabilities)


if __name__ == '__main__':
    print("UP五星角色期望", up_5star_character(1).exp)
    print("UP五星光锥期望", up_5star_weapon(1).exp)
    # 计算分位点
    # quantile_pos = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    # print("选择分位点"+str(quantile_pos))
    # for c in range(0, 7):
    #     for w in range(0, 6):
    #         dist = up_5star_character(c) * up_5star_weapon(w)
    #         print(f"{c}魂{w}叠 "+str(dist.quantile_point(quantile_pos)))
