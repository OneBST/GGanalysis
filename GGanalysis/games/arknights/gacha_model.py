from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

'''
    注意，本模块按公示概率进行建模，但忽略了一些情况
        如不纳入对300保底的考虑，获取1个物品的分布不会在第300抽处截断
        这么做的原因是，模型只支持一抽最多获取1个物品，若在第300抽处刚好抽到物品
        等于一抽抽到两个物品，无法处理。
        对于这个问题，建议结合抽数再加一步分析，进行一次后处理（如使用 QuantileFunction 中的 direct_exchange 项）
'''

__all__ = [
    'PITY_6STAR',
    'PITY_5STAR',
    'P_5STAR_AVG',
    'common_6star',
    'single_up_6star_old',
    'single_up_6star',
    'dual_up_specific_6star_old',
    'dual_up_specific_6star',
    'limited_up_6star',
    'common_5star',
    'single_up_specific_5star',
    'dual_up_specific_5star',
    'triple_up_specific_5star',
    'limited_both_up_6star',
]

# 设置6星概率递增表
PITY_6STAR = np.zeros(100)
PITY_6STAR[1:51] = 0.02
PITY_6STAR[51:99] = np.arange(1, 49) * 0.02 + 0.02
PITY_6STAR[99] = 1
# 设置5星概率递增表（5星保底会被6星挤掉，所以需要做一点近似）
PITY_5STAR = np.zeros(42, dtype=float)
PITY_5STAR[:16] = 0.08
PITY_5STAR[16:21] = np.arange(1, 6) * 0.02 + 0.08
PITY_5STAR[21:41] = np.arange(1, 21) * 0.04 + 0.18
PITY_5STAR[41] = 1
# 设置5星综合概率，用于近似计算
P_5STAR_AVG = 0.08948


class AK_Limit_Model(CommonGachaModel):
    '''
    方舟限定池同时获取限定6星及陪跑6星模型
    
    - ``total_item_types`` 道具的总类别数
    - ``collect_item`` 收集道具的目标类别数
    '''

    def __init__(self, pity_p, p, total_item_types=2, collect_item=None, e_error=1e-8, max_dist_len=1e5) -> None:
        super().__init__()
        if collect_item is None:
            collect_item = total_item_types
        self.layers.append(PityLayer(pity_p))
        self.layers.append(BernoulliLayer(p, e_error, max_dist_len))
        self.layers.append(CouponCollectorLayer(total_item_types, collect_item))

    def __call__(self, multi_dist: bool = False, item_pity=0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(1, multi_dist, item_pity, *args, **kwds)

    def _build_parameter_list(self, item_pity: int = 0, ) -> list:
        parameter_list = [
            [[], {'item_pity': item_pity}],
            [[], {}],
            [[], {}]
        ]
        return parameter_list


class HardTypePityDP():
    '''
        类型硬保底DP（这里称原神的“平稳机制”为类型软保底，是一个类型的机制）
        描述超过一定抽数没有获取某类道具时，下次再获取道具时必为某类道具之一，直至某类道具获得完全的模型
        调用返回获得1个指定类型道具所需抽数分布
        TODO 稍加修改改为集齐k种道具所需的抽数分布
    '''

    def __init__(self, item_pull_dist, type_pity_gap, item_types=2, up_rate=1, type_pull_shift=0) -> None:
        '''以获取此道具抽数分布、类型保底抽数、道具类别数量初始化'''
        self.item_pull_dist = item_pull_dist
        self.item_pity_pos = len(self.item_pull_dist) - 1
        self.type_pity_gap = type_pity_gap
        self.item_types = item_types
        self.up_rate = up_rate
        self.type_pull_shift = type_pull_shift  # 用于处理 51 101 151 这类情况,shift=1

        # 输入检查
        if self.item_types <= 0:
            raise ValueError("item_types must above 0!")
        if self.item_pity_pos >= self.type_pity_gap:
            raise ValueError("Only support item_pity_pos < type_pity_pos")

    def __call__(self, item_pity=0, type_pity=0) -> np.ndarray:
        # 根据保底情况计算极限位置
        calc_pulls = len(
            self.item_pull_dist) - 1 + self.type_pity_gap * self.item_types - type_pity + self.type_pull_shift
        # 状态定义
        # 第0维为恰好在某一抽获得了道具，但一直没有目标类型
        # 第1维为是否获得目标类型
        M = np.zeros((calc_pulls + 1, 2), dtype=float)
        M[0, 0] = 1
        # 处理垫抽情况
        condition_dist = cut_dist(self.item_pull_dist, item_pity)
        # 开始DP 理论上用numpy矢量操作会快一点，但是这样不好写if就放弃了
        for i in range(1, calc_pulls + 1):
            # 枚举上次获得物品位置
            for j in range(max(0, i - self.item_pity_pos), i):
                # 有保底的情况下单独考虑
                use_dist = self.item_pull_dist
                if j == 0:
                    use_dist = condition_dist
                    if i - j >= len(use_dist):
                        continue
                        # 处理种类平稳情况，这里考虑的是倍数情况下
                if max(i - 1 + type_pity - self.type_pull_shift, 0) // self.type_pity_gap != max(
                        j - 1 + type_pity - self.type_pull_shift, 0) // self.type_pity_gap:
                    # 跨过平稳抽数的整倍数时触发了种类平稳，判断此时还没有抽到需求种类时转移的概率
                    type_p = (max(i - 1 + type_pity - self.type_pull_shift, 0) // self.type_pity_gap) / self.item_types
                    # print(j, i, type_p)
                else:
                    # 没触发种类平稳
                    type_p = self.up_rate / self.item_types
                M[i, 0] += M[j, 0] * (1 - type_p) * use_dist[i - j]
                M[i, 1] += M[j, 0] * type_p * use_dist[i - j]
        return M[:, 1]


class AKHardPityModel(CommonGachaModel):
    '''
    针对通过统计发现的类型保底
    
    该机制目前仅发现存在于标准寻访-双UP轮换池（不包含中坚寻访-双UP轮换池），尚未知该机制的累计次数是否会跨卡池继承, `详细信息参看一个资深的烧饼-视频 <https://www.bilibili.com/video/BV1ib411f7YF/>`_ '''

    def __init__(self, no_type_pity_dist: FiniteDist, item_pull_dist, type_pity_gap, item_types=2, up_rate=1,
                 type_pull_shift=0) -> None:
        super().__init__()
        self.DP_module = HardTypePityDP(item_pull_dist, type_pity_gap, item_types, up_rate, type_pull_shift)
        self.no_type_pity_dist = no_type_pity_dist

    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity=0, type_pity=0) -> Union[
        FiniteDist, list]:
        if not multi_dist:
            return FiniteDist(self.DP_module(item_pity, type_pity)) * self.no_type_pity_dist ** (item_num - 1)
        else:
            ans_list = [FiniteDist([1]), FiniteDist(self.DP_module(item_pity, type_pity))]
            for i in range(2, item_num + 1):
                ans_list.append(ans_list[i - 1] * self.no_type_pity_dist)
            return ans_list


class AKDirectionalModel(CommonGachaModel):
    '''
    针对描述的寻访规则调整中提到的 `定向选调机制 <https://www.bilibili.com/read/cv22596510>`_ （在这里称为类型保底）
    
    - ``type_pity_gap`` 类型保底触发间隔抽数
    - ``item_types`` 道具类别个数，可以简单理解为UP了几个干员
    - ``up_rate`` 道具所占比例
    - ``type_pull_shift`` 保底类型偏移量，默认为0（无偏移）
    '''

    def __init__(self, no_type_pity_dist: FiniteDist, item_pull_dist, type_pity_gap, item_types=2, up_rate=1,
                 type_pull_shift=0) -> None:
        super().__init__()
        self.DP_module = HardTypePityDP(item_pull_dist, type_pity_gap, item_types, up_rate, type_pull_shift)
        self.no_type_pity_dist = no_type_pity_dist
        self.type_pity_gap = type_pity_gap
        self.item_pull_dist = item_pull_dist
        self.no_type_pity_dist = no_type_pity_dist
        self.type_pull_shift = type_pull_shift

    def _get_dist(self, item_num, item_pity, type_pity):
        if item_num == 1:
            return FiniteDist(self.DP_module(item_pity, type_pity))
        # 第一个保底的概率分布 (没有归一化)
        c_dist = FiniteDist(self.DP_module(item_pity, type_pity))
        c_dist[:self.type_pity_gap + self.type_pull_shift - type_pity + 1] = 0
        # 其他保底的概率分布 (没有归一化)
        f_dist = FiniteDist(self.DP_module(0, 0))
        f_dist[:self.type_pity_gap + self.type_pull_shift + 1] = 0
        # 第一个不保底的概率分布 (没有归一化)
        first_lucky = FiniteDist(
            self.DP_module(item_pity, type_pity)[:self.type_pity_gap + self.type_pull_shift - type_pity + 1])
        # 随后不保底的概率分布 (没有归一化)
        rest_lucky = FiniteDist(self.no_type_pity_dist[:self.type_pity_gap + self.type_pull_shift + 1])

        ans = FiniteDist([0])
        # 处理第一个就保底
        ans += (c_dist * self.no_type_pity_dist ** (item_num - 1))
        # 处理其他情况
        for i in range(2, item_num + 1):
            ans += (first_lucky * rest_lucky ** (i - 2) * f_dist * self.no_type_pity_dist ** (item_num - i))
        # 处理一直不保底的情况
        ans += first_lucky * rest_lucky ** (item_num - 1)
        return ans
    
    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity=0, type_pity=0) -> Union[
        FiniteDist, list]:
        if not multi_dist:
            return self._get_dist(item_num, item_pity, type_pity)
        else:
            ans_list = [FiniteDist([1])]
            for i in range(1, item_num + 1):
                ans_list.append(self._get_dist(i, item_pity, type_pity))
            return ans_list


# ★★★★★
# 5星公示概率为的8%，实际上综合概率为8.948% 这里按照综合概率近似计算
# 注意，这里的5星并没有考虑类型保底和概率上升的情况，是非常简单的近似，具体类型保底机制详情见 https://www.bilibili.com/opus/772396224252739622
# 获取普通5星
common_5star = BernoulliGachaModel(P_5STAR_AVG)
# 获取单UP5星
single_up_specific_5star = BernoulliGachaModel(P_5STAR_AVG / 2)
# 获取双UP5星中的特定5星
dual_up_specific_5star = BernoulliGachaModel(P_5STAR_AVG / 2 / 2)
# 获取三UP5星中的特定5星
triple_up_specific_5star = BernoulliGachaModel(P_5STAR_AVG * 0.6 / 3)

# ★★★★★★
# 注意有定向选调的情况只处理了第一个，接下来的没有处理，是有缺陷的，之后需要重写DP
# 获取普通6星
common_6star = PityModel(PITY_6STAR)
# 获取单UP6星 无定向选调及有定向选调 定向选调见 https://www.bilibili.com/read/cv22596510/
single_up_6star_old = PityBernoulliModel(PITY_6STAR, 1 / 2)
single_up_6star = AKDirectionalModel(single_up_6star_old(1), p2dist(PITY_6STAR), type_pity_gap=150, item_types=1,
                                     up_rate=0.5)
# 获取双UP6星中的特定6星 无类型保底及有类型保底，此抽卡机制尚未完全验证
dual_up_specific_6star_old = PityBernoulliModel(PITY_6STAR, 1 / 4)
# 据统计 https://www.bilibili.com/video/BV1ib411f7YF/ 此卡池保底实际上从 202/402 等位置开始触发，因此此处 ``type_pull_shift`` 填写为 1
dual_up_specific_6star = AKHardPityModel(dual_up_specific_6star_old(1), p2dist(PITY_6STAR), type_pity_gap=200,
                                         item_types=2, up_rate=0.5, type_pull_shift=1)
# 获取限定UP6星中的限定6星
limited_up_6star = PityBernoulliModel(PITY_6STAR, 0.35)
# 同时获取限定池中两个UP6星（没有考虑井，不适用于有定向选调的卡池）
limited_both_up_6star = AK_Limit_Model(PITY_6STAR, 0.7, total_item_types=2, collect_item=2)

if __name__ == '__main__':
    pass
