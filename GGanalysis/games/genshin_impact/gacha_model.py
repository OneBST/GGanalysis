'''
    注意，本模块对4星概率进行了近似处理
        1. 仅考虑不存在5星物品时的情况，没有考虑4星物品被5星物品挤到下一抽的可能
        2. 对于UP4星物品，没有考虑UP4星从常驻中也有概率获取的可能性
    计算所得4星综合概率会略高于实际值，获取UP4星的概率略低于实际值，但影响非常微弱可以忽略
    
    同时由于复杂性，原神的平稳机制没有纳入计算，其影响也很低。如果想了解平稳机制的影响，可以使用 GGanalysislib 工具包
    见 https://github.com/OneBST/GGanalysis
'''
from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

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
    'up_5star_specific_weapon',
    'up_4star_weapon',
    'up_4star_specific_weapon',
    'up_5star_ep_weapon',
    'stander_5star_character_in_up',
    'stander_5star_weapon_in_up',
]

# 原神普通5星保底概率表
PITY_5STAR = np.zeros(91)
PITY_5STAR[1:74] = 0.006
PITY_5STAR[74:90] = np.arange(1, 17) * 0.06 + 0.006
PITY_5STAR[90] = 1
# 原神普通4星保底概率表
PITY_4STAR = np.zeros(11)
PITY_4STAR[1:9] = 0.051
PITY_4STAR[9] = 0.051 + 0.51
PITY_4STAR[10] = 1
# 原神武器池5星保底概率表
PITY_W5STAR = np.zeros(78)
PITY_W5STAR[1:63] = 0.007
PITY_W5STAR[63:77] = np.arange(1, 15) * 0.07 + 0.007
PITY_W5STAR[77] = 1
# 原神武器池4星保底概率表
PITY_W4STAR = np.zeros(10)
PITY_W4STAR[1:8] = 0.06
PITY_W4STAR[8] = 0.06 + 0.6
PITY_W4STAR[9] = 1

# 定义获取星级物品的模型
common_5star = PityModel(PITY_5STAR)
common_4star = PityModel(PITY_4STAR)
# 定义原神角色池模型
up_5star_character = DualPityModel(PITY_5STAR, [0, 0.5, 1])
up_4star_character = DualPityModel(PITY_4STAR, [0, 0.5, 1])
up_4star_specific_character = DualPityBernoulliModel(PITY_4STAR, [0, 0.5, 1], 1/3)
# 定义原神武器池模型
common_5star_weapon = PityModel(PITY_W5STAR)
common_4star_weapon = PityModel(PITY_W4STAR)
up_5star_weapon = DualPityModel(PITY_W5STAR, [0, 0.75, 1])
up_5star_specific_weapon = DualPityBernoulliModel(PITY_W5STAR, [0, 0.75, 1], 1/2)
up_4star_weapon = DualPityModel(PITY_W4STAR, [0, 0.75, 1])
up_4star_specific_weapon = DualPityBernoulliModel(PITY_W4STAR, [0, 0.75, 1], 1/5)

# 定轨获取特定UP5星武器
class Genshin5starEPWeaponModel(CommonGachaModel):
    def __init__(self) -> None:
        super().__init__()
        self.state_num = {'get':0, 'fate0pity':1, 'fate1':2, 'fate1pity':3, 'fate2':4}
        state_trans = [
            ['get', 'get', 0.375],
            ['get', 'fate1', 0.375],
            ['get', 'fate1pity', 0.25],
            ['fate0pity', 'get', 0.5],
            ['fate0pity', 'fate1', 0.5],
            ['fate1', 'get', 0.375],
            ['fate1', 'fate2', 1-0.375],
            ['fate1pity', 'get', 0.5],
            ['fate1pity', 'fate2', 0.5],
            ['fate2', 'get', 1]
        ]
        M = table2matrix(self.state_num, state_trans)
        self.layers.append(PityLayer(PITY_W5STAR))
        self.layers.append(MarkovLayer(M))

    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity = 0, up_pity = 0, fate_point = 0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, up_pity, fate_point, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0, up_pity: int=0, fate_point: int=0) -> list:
        if fate_point >= 2:
            begin_pos = 4
        elif fate_point == 1 and up_pity == 1:
            begin_pos = 3
        elif fate_point == 1 and up_pity == 0:
            begin_pos = 2
        elif fate_point == 0 and up_pity == 1:
            begin_pos = 1
        else:
            begin_pos = 0
        l1_param = [[], {'item_pity':item_pity}]
        l2_param = [[], {'begin_pos':begin_pos}]
        parameter_list = [l1_param, l2_param]
        return parameter_list

up_5star_ep_weapon = Genshin5starEPWeaponModel()


class GenshinCommon5starInUPpoolModel(CommonGachaModel):
    def __init__(self, up_rate=0.5, stander_item=7, dp_lenth=500, need_type=1, max_dist_len=1e5) -> None:
        super().__init__()
        self.layers.append(PityLayer(PITY_5STAR))
        self.layers.append(GenshinCommon5starInUPpoolLayer(up_rate, stander_item, dp_lenth, need_type, max_dist_len))
    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity = 0, is_last_UP=False, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, is_last_UP, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0, is_last_UP: bool=False) -> list:
        l1_param = [[], {'item_pity':item_pity}]
        l2_param = [[], {'is_last_UP':is_last_UP}]
        parameter_list = [l1_param, l2_param]
        return parameter_list

class GenshinCommon5starInUPpoolLayer(GachaLayer):
    # 原神常驻歪UP层 修改自 PityLayer
    def __init__(self, up_rate=0.5, stander_item=7, dp_lenth=500, need_type=1, max_dist_len=1e5) -> None:
        super().__init__()
        self.up_rate = up_rate
        self.stander_item = stander_item
        self.dp_lenth = dp_lenth  # DP的截断位置，越长计算误差越小，500时误差可以忽略了
        self.need_type = min(need_type, self.stander_item)  # 需要的5星类型
        self.max_dist_len = max_dist_len # 返回单道具的长度极限，切断后可以省计算量

    def calc_5star_number_dist(self, is_last_UP=False):
        # DP数组 表示抽第i个5星时
        # 恰好抽到UP物品、非目标其他物品、目标其他物品的概率
        M = np.zeros((self.dp_lenth+1, 3), dtype=float)
        if is_last_UP: # 从才获取了UP开始，下一个就可以是非UP
            M[0][0] = 1
        else:
            M[0][1] = 1  # 从才获取了非UP开始
        for i in range(1, self.dp_lenth+1):
            M[i][0] = self.up_rate * M[i-1][0] + M[i-1][1]
            M[i][1] = (1 - self.up_rate) * (self.stander_item-self.need_type)/self.stander_item * M[i-1][0]
            M[i][2] = (1 - self.up_rate) * self.need_type/self.stander_item * M[i-1][0]
        # 截断位置补概率，保证归一化
        M[0][2] = 0
        M[self.dp_lenth][2] = 1 - np.sum(M[:self.dp_lenth, 2])
        return FiniteDist(M[:, 2])
    
    def __str__(self) -> str:
        return f"GenshinCommon5starInUPpoolLayer UP rate={round(self.up_rate, 2)} Stander Item={self.stander_item} DP Lenth={self.dp_lenth}"
    def _forward(self, input, full_mode, is_last_UP=False) -> FiniteDist:
        # 输入为空，本层为第一层，返回初始分布
        if input is None:
            return self.calc_5star_number_dist(is_last_UP)
        # 处理累加分布情况
        # f_dist 为完整分布 c_dist 为条件分布 根据工作模式不同进行切换
        f_dist: FiniteDist = input[0]
        if full_mode:
            c_dist: FiniteDist = input[0]
        else:
            c_dist: FiniteDist = input[1]
        # 处理条件叠加分布
        if full_mode:
            # 返回完整分布 从才获取了非UP开始
            overlay_dist = self.calc_5star_number_dist(False)
        else:
            # 根据参数，返回条件分布
            overlay_dist = self.calc_5star_number_dist(is_last_UP)

        # 以下全部来自 PityLayer 2023-02-18
        output_dist = FiniteDist([0])  # 获得一个0分布
        output_E = 0  # 叠加后的期望
        output_D = 0  # 叠加后的方差
        temp_dist = FiniteDist([1]) # 用于优化计算量
        for i in range(1, len(overlay_dist)):
            c_i = float(overlay_dist[i])  # 防止类型错乱的缓兵之策 如果 c_i 的类型是 numpy 数组，则 numpy 会接管 finite_dist_1D 定义的运算返回错误的类型
            # output_dist += c_i * (c_dist * f_dist ** (i-1))  # 分布累加
            # 修改一下优化计算量
            output_dist += c_i * (c_dist * temp_dist)  # 分布累加
            temp_dist = temp_dist * f_dist
            output_E += c_i * (c_dist.exp + (i-1) * f_dist.exp)  # 期望累加
            output_D += c_i * (c_dist.var + (i-1) * f_dist.var + (c_dist.exp + (i-1) * f_dist.exp) ** 2)  # 期望平方累加
        output_D -= output_E ** 2  # 计算得到方差
        output_dist.exp = output_E
        output_dist.var = output_D
        if len(output_dist) > self.max_dist_len:
            output_dist.dist = output_dist.dist[:int(self.max_dist_len)]
        return output_dist
    
stander_5star_character_in_up = GenshinCommon5starInUPpoolModel(up_rate=0.5, stander_item=7, dp_lenth=300, need_type=1)
stander_5star_weapon_in_up = GenshinCommon5starInUPpoolModel(up_rate=0.75, stander_item=10, dp_lenth=800, need_type=1)


if __name__ == '__main__':
    pass