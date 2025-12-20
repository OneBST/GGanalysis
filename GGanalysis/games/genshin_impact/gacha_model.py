'''
    注意，本模块对4星概率进行了近似处理
        1. 仅考虑不存在5星物品时的情况，没有考虑4星物品被5星物品挤到下一抽的可能
        2. 对于UP4星物品，没有考虑UP4星从常驻中也有概率获取的可能性
    计算所得4星综合概率会略高于实际值，获取UP4星的概率略低于实际值，但影响非常微弱可以忽略
    
    捕获明光机制还没有完全解析，模型计算出来的所需抽数会比实际更高

    同时由于复杂性，原神的平稳机制没有纳入计算，其影响也很低。如果想了解平稳机制的影响，可以使用 GGanalysislib 工具包
    见 https://github.com/OneBST/GGanalysis
'''
from GGanalysis.distribution_1d import *
from GGanalysis.markov_method import table2matrix
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

__all__ = [
    'PITY_5STAR',
    'PITY_4STAR',
    'PITY_W5STAR',
    'PITY_W4STAR',
    'CR_P',

    'common_5star',
    'common_4star',
    'up_5star_character',
    'up_4star_character',
    'up_4star_specific_character',
    'common_5star_weapon',
    'common_4star_weapon',
    'up_5star_weapon',
    'up_5star_ep_weapon',
    'up_4star_weapon',
    'up_4star_specific_weapon',
    
    'classic_stander_5star_character_in_up',
    'classic_stander_5star_weapon_in_up',
    'classic_up_5star_character',
    'classic_up_5star_ep_weapon',
    'classic_up_5star_specific_weapon',

    'ClassicGenshinCommon5starInUPpoolModel',
    'CapturingRadianceModel',
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
# 捕获明光计数器模型触发概率，此处定义为触发明光概率P，非等效UP概率。等效UP概率为 P+(1-P)/2=0.5+P/2
CR_P = [0, 0, 0, 1]

# 5.0前命定值为2的定轨获取特定UP5星武器
class ClassicGenshin5starEPWeaponModel(CommonGachaModel):
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

    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity = 0, up_pity = 0, fate_point = 0) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, up_pity, fate_point)

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

class ClassicGenshinCommon5starInUPpoolLayer(GachaLayer):
    # 原神UP池获得特定常驻角色DP 修改自 PityLayer 适用于捕获明光机制出现前
    def __init__(self, up_rate=0.5, stander_item=8, dp_lenth=500, need_type=1, max_dist_len=1e5) -> None:
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
    
    @lru_cache
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
        # 对0位置特殊处理
        output_dist += float(overlay_dist[0]) * temp_dist
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
            output_dist.__dist = output_dist.__dist[:int(self.max_dist_len)]
        return output_dist

class ClassicGenshinCommon5starInUPpoolModel(CommonGachaModel):
    # 原神UP池获得特定常驻角色模型 适用于捕获明光机制出现前
    def __init__(self, up_rate=0.5, stander_item=8, dp_lenth=500, need_type=1, max_dist_len=1e5) -> None:
        super().__init__()
        self.layers.append(PityLayer(PITY_5STAR))
        self.layers.append(ClassicGenshinCommon5starInUPpoolLayer(up_rate, stander_item, dp_lenth, need_type, max_dist_len))
    
    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity = 0, is_last_UP=False) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, is_last_UP)

    def _build_parameter_list(self, item_pity: int=0, is_last_UP: bool=False) -> list:
        l1_param = [[], {'item_pity':item_pity}]
        l2_param = [[], {'is_last_UP':is_last_UP}]
        parameter_list = [l1_param, l2_param]
        return parameter_list
    
def capturing_radiance_dp(item_num=1, up_pity=0, cr_count=1, cr_p=CR_P):
    # 估计值，是上限，有的情况不会到达
    max_5star = (item_num // 3 * 5) + item_num % 3 * 2 + int(cr_count==0)
    # (获取了n个五星时恰好获得了，第m个UP五星，此时的计数器值)
    M = np.zeros((max_5star+1, item_num+1, 4), dtype=float)
    # 初始值 分别为当前已经使用了多少个五星，当前获得UP五星数量，当前计数器值
    M[up_pity,up_pity,cr_count] = 1
    for i in range(1, max_5star+1):
        for j in range(1, item_num+1):
            # 本次通过大保底获得道具
            if i >= 2:
                for k in range(1,4):
                    M[i,j,k] += M[i-2,j-1,k-1] * (0.5 - cr_p[k-1]/2)
            # 本次小保底获得道具
            for k in range(0,2):
                M[i,j,k] += M[i-1,j-1,k+1] * (0.5 - cr_p[k+1]/2)
            M[i,j,0] += M[i-1,j-1,0] * (0.5 - cr_p[0]/2)
            # 本次触发捕获明光获得道具，默认回到计数器为1状态
            for k in range(0,4):
                M[i,j,1] += M[i-1,j-1,k] * cr_p[k]
    # 返回消耗五星分布
    return np.trim_zeros(np.sum(M[:, item_num, :], axis=1), 'b')

class CapturingRadianceModel(GachaModel):
    '''
    针对原神5.0后加入的「捕获明光」机制的计数器模型
    模型不完全完善，解析见 https://www.bilibili.com/video/BV13XBiYZErT/
    '''
    def __init__(self, pity5_p=PITY_5STAR, cr_p=CR_P) -> None:
        self.common_5star = PityModel(pity5_p)
        self.cr_p = cr_p
    def _get_cr_5star_dist(self, item_num, up_pity=0, cr_counter=0):
        return FiniteDist(capturing_radiance_dp(item_num, up_pity, cr_counter, self.cr_p))
    def _get_dist(self, item_num, item_pity, up_pity, cr_counter):
        # 计算抽五星所需的抽数分布
        f_dist = self.common_5star(1)
        c_dist = self.common_5star(1, item_pity=item_pity)
        # 「捕获明光」机制下抽 item_num 个UP五星所需的五星层数，使用分布列定义抽卡层
        cr_layer = PityLayer(self._get_cr_5star_dist(item_num, up_pity, cr_counter))
        ans = cr_layer._forward((f_dist, c_dist), False, 0)
        return ans

    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity=0, up_pity=0, cr_counter=1) -> Union[
        FiniteDist, list]:
        if up_pity and cr_counter==0:
            raise ValueError("up_pity 数值与 cr_counter 数值矛盾")
        if not multi_dist:
            return self._get_dist(item_num, item_pity, up_pity, cr_counter)
        else:
            ans_list = [FiniteDist([1])]
            for i in range(1, item_num + 1):
                ans_list.append(self._get_dist(i, item_pity, up_pity, cr_counter))
            return ans_list

# 定义获取星级物品的模型
common_5star = PityModel(PITY_5STAR)
common_4star = PityModel(PITY_4STAR)
# 定义原神角色池模型
classic_up_5star_character = DualPityModel(PITY_5STAR, [0, 0.5, 1])
up_5star_character = CapturingRadianceModel(PITY_5STAR)
up_4star_character = DualPityModel(PITY_4STAR, [0, 0.5, 1])
up_4star_specific_character = DualPityBernoulliModel(PITY_4STAR, [0, 0.5, 1], 1/3)
# 定义原神武器池模型
common_5star_weapon = PityModel(PITY_W5STAR)
common_4star_weapon = PityModel(PITY_W4STAR)
up_5star_weapon = DualPityModel(PITY_W5STAR, [0, 0.75, 1])
up_5star_ep_weapon = DualPityModel(PITY_W5STAR, [0, 0.375, 1])  # 5.0后命定值为1的有定轨武器池
classic_up_5star_ep_weapon = ClassicGenshin5starEPWeaponModel()  # 2.0后至5.0前命定值为2的有定轨武器池
classic_up_5star_specific_weapon = DualPityBernoulliModel(PITY_W5STAR, [0, 0.75, 1], 1/2)  # 2.0前无定轨的武器池
up_4star_weapon = DualPityModel(PITY_W4STAR, [0, 0.75, 1])
up_4star_specific_weapon = DualPityBernoulliModel(PITY_W4STAR, [0, 0.75, 1], 1/5)
# 5.0前从UP池中获取常驻角色计算
classic_stander_5star_character_in_up = ClassicGenshinCommon5starInUPpoolModel(up_rate=0.5, stander_item=7, dp_lenth=300, need_type=1)
classic_stander_5star_weapon_in_up = ClassicGenshinCommon5starInUPpoolModel(up_rate=0.75, stander_item=10, dp_lenth=800, need_type=1)

if __name__ == '__main__':
    print(up_5star_character(1, cr_counter=3).exp)
    print(classic_up_5star_specific_weapon(1).exp)
    print(common_5star(1).exp)
    print(common_5star_weapon(1).exp)
    pass