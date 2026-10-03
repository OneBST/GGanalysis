'''
    注意，本模块对4星概率进行了近似处理
        1. 仅考虑不存在5星物品时的情况，没有考虑4星物品被5星物品挤到下一抽的可能
        2. 对于UP4星物品，没有考虑UP4星从常驻中也有概率获取的可能性
    计算所得4星综合概率会略高于实际值，获取UP4星的概率略低于实际值，但影响非常微弱可以忽略
    
    捕获明光采用双计数器猜想，B 计数器触发概率为拟合值，不代表已确认机制。
    常驻五星角色/武器类别入口包含平稳机制；四星仍使用上述近似。
'''
from GGanalysis.distribution_1d import *
from functools import cached_property
from GGanalysis.markov import (
    StateSpace, TransitionBuilder, HitTransitionBuilder, HitProcessAnalysis,
    ChainAnalysis, StateRewards,
)
from GGanalysis.markov.transition import validate_steps
from .standard_pity import StandardGenshin5starModel
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

__all__ = [
    'PITY_5STAR',
    'PITY_4STAR',
    'PITY_W5STAR',
    'PITY_W4STAR',
    'CR_P',
    'CR_B_P',

    'common_5star',
    'common_4star',
    'standard_5star_character',
    'standard_5star_weapon',
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
    'StandardGenshin5starModel',
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
# 双计数器猜想：B 在 0～5 不触发，6～9 使用拟合值，10 起必触发。
CR_B_P = [0, 0, 0, 0, 0, 0, 0.01, 0.05, 0.25, 0.99, 1]

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
        builder = TransitionBuilder(StateSpace.from_shape([len(self.state_num)]), backend="dense")
        for source, destination, probability in state_trans:
            builder.add(self.state_num[source], self.state_num[destination], probability)
        M = builder.build(check=True).P
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
        if len(output_dist) > self.max_dist_len:
            max_length = int(self.max_dist_len)
            truncated = output_dist.dist[:max_length]
            output_dist.set_dist(
                truncated,
                trim_tail_zeros=False,
                exp=output_E,
                var=output_D,
                tail_mass=max(0.0, 1.0 - float(np.sum(truncated))),
                metadata={
                    'method': 'truncated',
                    'truncation_length': max_length,
                    'moment_source': 'theoretical',
                },
            )
        else:
            output_dist.exp = output_E
            output_dist.var = output_D
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
    
class CapturingRadianceModel(GachaModel):
    """捕获明光双计数器猜想：五星层上的状态为 (A,B,大保底)。

    默认初态 A=1、B=3。小保底的直接明光概率取 cr_p[A] 与 cr_b_p[B]
    较大者；默认 A 仅在 3 时强制触发，其他情况只由 B 控制。
    未触发明光的剩余概率均分给普通不歪和歪；不是歪后救回概率。
    普通不歪令 A-1、B-2（最低零），歪令 A+1、B+3；明光重置至 (1,3)。
    大保底仅清除保证标记，不更新 A、B。B 参数为用户提供的拟合猜想。

    pity5_p 是五星保底概率表；cr_p 长度为 4，cr_b_p 长度为 11，
    最后一项均须为 1。B>=10 直接明光；歪后暂存的 B 最大为 12。
    """
    def __init__(self, pity5_p=PITY_5STAR, cr_p=CR_P, *, cr_b_p=CR_B_P) -> None:
        self.common_5star = PityModel(pity5_p)
        self.cr_p = np.array(cr_p, dtype=float, copy=True)
        self.cr_b_p = np.array(cr_b_p, dtype=float, copy=True)
        for table, length in ((self.cr_p, 4), (self.cr_b_p, 11)):
            if (table.shape != (length,) or not np.all(np.isfinite(table)) or
                    np.any(table < 0) or np.any(table > 1) or table[-1] != 1):
                raise ValueError("明光概率表须为指定长度的概率向量，末项为 1。")
            table.flags.writeable = False

    @cached_property
    def _analysis(self) -> HitProcessAnalysis:
        full = StateSpace.from_shape([4, 13, 2])
        boundary = StateSpace.from_shape([4, 13])
        embed = [full.state_to_id([a, b, 0]) for a in range(4) for b in range(13)]
        builder = HitTransitionBuilder(full, boundary, embed)
        for a in range(4):
            for b in range(13):
                source = full.state_to_id([a, b, 0])
                c = max(self.cr_p[a], self.cr_b_p[min(b, 10)])
                builder.add_hit(source, boundary.state_to_id([1, 3]), c)
                builder.add_hit(source, boundary.state_to_id([max(a-1, 0), max(b-2, 0)]), (1-c)/2)
                if c < 1:
                    builder.add_miss(source, full.state_to_id([a+1, b+3, 1]), (1-c)/2)
                builder.add_hit(full.state_to_id([a, b, 1]), boundary.state_to_id([a, b]), 1)
        return HitProcessAnalysis(builder.build(check=True), max_steps=2)

    def _initial(self, up_pity: int, cr_counter: int, cr_counter_b: int) -> np.ndarray:
        validate_steps(cr_counter)
        validate_steps(cr_counter_b)
        if up_pity not in (0, 1) or cr_counter > 3 or cr_counter_b > 12:
            raise ValueError("up_pity 为 0/1，A 为 0～3，B 为 0～12。")
        if up_pity and cr_counter == 0:
            raise ValueError("up_pity 数值与 cr_counter 数值矛盾")
        return self._analysis.process.full_space.delta([cr_counter, cr_counter_b, int(up_pity)])

    def _get_cr_5star_dist(self, item_num: int, up_pity: int = 0,
                          cr_counter: int = 1, cr_counter_b: int = 3) -> FiniteDist:
        return self._analysis.nth_state_dist(
            item_num, self._initial(up_pity, cr_counter, cr_counter_b), method="direct",
        ).marginal_cost()

    def _to_pulls(self, count_dist: FiniteDist, item_pity: int) -> FiniteDist:
        # 五星等待周期 IID，只有首个五星使用当前水位的条件分布。
        f_dist = self.common_5star(1)
        c_dist = self.common_5star(1, item_pity=item_pity)
        return PityLayer(count_dist)._forward((f_dist, c_dist), False, 0)

    @cached_property
    def _small_pity_up_rate(self) -> float:
        process = self._analysis.process
        chain = ChainAnalysis(process.transition)
        stationary = chain.long_run_state(self._initial(0, 1, 3))
        small = (process.full_space.ids_to_states(np.arange(process.full_space.N))[:, 2] == 0)
        up = np.asarray(process.hit.sum(axis=0)).ravel()
        rewards = StateRewards(process.full_space, {"small": small, "small_up": small*up})
        values = rewards.expectation(stationary)
        return values["small_up"] / values["small"]

    def small_pity_up_rate(self) -> float:
        """返回长期小保底 UP 胜率，不计大保底；默认拟合参数约为 0.5526916771。

        为初态 (A=1,B=3) 所在长期类的结果，不是指定当前计数器的下一次胜率。
        若需含大保底的五星 UP 比例，可用 1/(2-p) 转换本返回值 p。
        """
        return self._small_pity_up_rate

    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity: int = 0,
                 up_pity: int = 0, cr_counter: int = 1, cr_counter_b: int = 3) -> Union[FiniteDist, list]:
        """返回获得 item_num 件 UP 的抽数分布，cr_counter 和 cr_counter_b 分别为 A、B。

        item_pity 为五星水位，up_pity=1 为大保底。multi_dist=True 返回
        0～item_num 件的分布列表；零件沿用旧接口，直接返回零花费 FiniteDist。
        首次与后续 UP 之间保留 A、B 状态，不假设 UP 获得周期 IID。
        """
        validate_steps(item_num)
        if item_num == 0:
            return FiniteDist([1])
        initial = self._initial(up_pity, cr_counter, cr_counter_b)
        if not multi_dist:
            joint = self._analysis.nth_state_dist(item_num, initial, method="direct")
            return self._to_pulls(joint.marginal_cost(), item_pity)
        return [FiniteDist([1])] + [self._to_pulls(joint.marginal_cost(), item_pity)
                                   for joint in self._analysis.iter_state_dists(item_num, initial, method="direct")]

class EpitomizedPathModel(GachaModel):
    '''
    针对原神5.0后武器池命定值从2变为1的改变
    设置了关于UP大小保底的选项
    '''
    def __init__(self, pity_p1, pity_p2, pity_p3):
        super().__init__()
        self.base_model = DualPityModel(pity_p1, pity_p2)
        self.up_pity_model = DualPityModel(pity_p1, pity_p3)

    def __call__(self, item_num: int=1, multi_dist: bool=False, item_pity=0, ep_pity=0, up_pity=0) -> Union[FiniteDist, list]:
        '''
        抽取个数 是否要返回多个分布列 道具保底进度 单次赠送保底进度
        '''
        # 处理没有武器大保底的情况
        if (not up_pity) or (ep_pity == 1):
            return self.base_model(item_num, multi_dist, item_pity, ep_pity)
        if item_num == 0:
            return FiniteDist([1])
        # 如果 multi_dist 参数为真，返回抽取 [1, 抽取个数] 个道具的分布列表
        if multi_dist:
            return self._get_multi_dist(item_num, item_pity)
        # 其他情况正常返回
        return self._get_dist(item_num, item_pity)
    
    # 输入 [完整分布, 条件分布] 指定抽取个数，返回抽取 [1, 抽取个数] 个道具的分布列表
    def _get_multi_dist(self, item_num: int, item_pity):
        # 仿造 CommonGachaModel 里的实现
        first_dist = self.up_pity_model(1, False, item_pity, 0)
        ans_list = [FiniteDist([1]), first_dist]
        if item_num > 1:
            # 处理剩余
            stander_dist = self.base_model(1)
            for i in range(1, item_num+1):
                ans_list.append(ans_list[i] * stander_dist)
        return ans_list
    
    # 返回单个分布
    def _get_dist(self, item_num: int, item_pity):
        first_dist = self.up_pity_model(1, False, item_pity, 0)
        if item_num == 1:
            return first_dist
        return first_dist * self.base_model(item_num-1)

# 定义获取星级物品的模型
common_5star = PityModel(PITY_5STAR)
common_4star = PityModel(PITY_4STAR)
# 定义原神角色池模型
classic_up_5star_character = DualPityModel(PITY_5STAR, [0, 0.5, 1])
up_5star_character = CapturingRadianceModel(PITY_5STAR)
standard_5star_character = StandardGenshin5starModel(PITY_5STAR, "character")
standard_5star_weapon = StandardGenshin5starModel(PITY_5STAR, "weapon")
up_4star_character = DualPityModel(PITY_4STAR, [0, 0.5, 1])
up_4star_specific_character = DualPityBernoulliModel(PITY_4STAR, [0, 0.5, 1], 1/3)
# 定义原神武器池模型
common_5star_weapon = PityModel(PITY_W5STAR)
common_4star_weapon = PityModel(PITY_W4STAR)
up_5star_weapon = DualPityModel(PITY_W5STAR, [0, 0.75, 1])
up_5star_ep_weapon_old = DualPityModel(PITY_W5STAR, [0, 0.375, 1])
up_5star_ep_weapon = EpitomizedPathModel(PITY_W5STAR, [0, 0.375, 1], [0, 0.5, 1])

classic_up_5star_ep_weapon = ClassicGenshin5starEPWeaponModel()  # 2.0后至5.0前命定值为2的有定轨武器池
classic_up_5star_specific_weapon = DualPityBernoulliModel(PITY_W5STAR, [0, 0.75, 1], 1/2)  # 2.0前无定轨的武器池
up_4star_weapon = DualPityModel(PITY_W4STAR, [0, 0.75, 1])
up_4star_specific_weapon = DualPityBernoulliModel(PITY_W4STAR, [0, 0.75, 1], 1/5)
# 5.0前从UP池中获取常驻角色计算
classic_stander_5star_character_in_up = ClassicGenshinCommon5starInUPpoolModel(up_rate=0.5, stander_item=7, dp_lenth=300, need_type=1)
classic_stander_5star_weapon_in_up = ClassicGenshinCommon5starInUPpoolModel(up_rate=0.75, stander_item=10, dp_lenth=800, need_type=1)

if __name__ == '__main__':
    # print(up_5star_character(1, cr_counter=3).exp)
    # print(classic_up_5star_specific_weapon(1).exp)
    # print(common_5star(1).exp)
    # print(common_5star_weapon(1).exp*(2-0.375))
    
    item_num = 2
    item_pity = 0
    ep_pity = 1

    print(up_5star_ep_weapon(item_num, item_pity=item_pity).exp)
    print(up_5star_ep_weapon(item_num, item_pity=item_pity, ep_pity=ep_pity, up_pity=0).exp)
    print(up_5star_ep_weapon_old(item_num, item_pity=item_pity, up_pity=ep_pity).exp)
    print(up_5star_ep_weapon(item_num, item_pity=item_pity, ep_pity=ep_pity, up_pity=1).exp)
