'''
    注意，本模块使用概率模型仅为根据游戏测试阶段推测，不能保证完全准确
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
    'common_5star_weapon',
    'common_4star_weapon',
    'up_5star_character',
    'pick_up_5star_character',
    'up_4star_character',
    'up_4star_specific_character',
    'up_5star_weapon',
    'pick_up_5star_weapon',
    'up_4star_weapon',
    'up_4star_specific_weapon',

    'ExclusiveRescreeningModel',
]

class ExclusiveRescreeningModel(GachaModel):
    '''
    2.5版本上线的打折复刻调频「独家重映」
    首次获取的S级代理人必为定向代理人
    首次10连八折
    暂且认为大保底会继承到下个卡池
    '''
    def __init__(self, first_model, afterwards_model):
        super().__init__()
        self.first_model = first_model
        self.afterwards_model = afterwards_model

    def _calc_first_ten_discount(self, dist: FiniteDist, discount=2):
        # 处理首次十连八折
        cdf = dist.cdf
        if len(cdf) < 10:
            # 处理长度小于10的部分
            cdf = np.zeros(11, dtype=float)
            cdf[10] = 1
        cdf[:10] = 0  # 一开始十连抽，1-9不可能出
        cdf = cdf[discount:]  # 折扣部分
        return cdf2dist(cdf)

    def __call__(self, item_num: int=1, multi_dist: bool=False, item_pity=0, up_pity=0, first_ten_discount=False) -> Union[FiniteDist, list]:
        '''
        抽取个数 是否要返回多个分布列 道具保底进度 单次赠送保底进度
        '''
        if item_num == 0:
            return FiniteDist([1])
        # 如果 multi_dist 参数为真，返回抽取 [1, 抽取个数] 个道具的分布列表
        if multi_dist:
            return self._get_multi_dist(item_num, item_pity, up_pity, first_ten_discount)
        # 其他情况正常返回
        return self._get_dist(item_num, item_pity, up_pity, first_ten_discount)
    
    # 输入 [完整分布, 条件分布] 指定抽取个数，返回抽取 [1, 抽取个数] 个道具的分布列表
    def _get_multi_dist(self, item_num: int, item_pity, up_pity, first_ten_discount):
        # 仿造 CommonGachaModel 里的实现
        first_dist = self.first_model(1, item_pity=item_pity)
        ans_list = [FiniteDist([1]), first_dist]
        if item_num > 1:
            # 处理第二个
            second_dist = self.afterwards_model(1, up_pity) * first_dist
            ans_list.append(second_dist)
        if item_num > 2:
            # 处理第三及更多个
            stander_dist = self.afterwards_model(1)
            for i in range(2, item_num):
                ans_list.append(ans_list[i] * stander_dist)
        if first_ten_discount:
            discount_list = [FiniteDist([1])]
            for i in range(1, item_num+1):
                discount_list.append(self._calc_first_ten_discount(ans_list[i]))
            return discount_list
        return ans_list
    
    # 返回单个分布
    def _get_dist(self, item_num: int, item_pity, up_pity, first_ten_discount):
        first_dist = self.first_model(1, item_pity=item_pity)
        if item_num == 1:
            if first_ten_discount:
                first_dist = self._calc_first_ten_discount(first_dist)
            return first_dist
        if first_ten_discount:
            return self._calc_first_ten_discount(first_dist * self.afterwards_model(item_num-1, up_pity))
        return first_dist * self.afterwards_model(item_num-1, up_pity)

# 绝区零普通5星保底概率表
PITY_5STAR = np.zeros(91)
PITY_5STAR[1:74] = 0.006
PITY_5STAR[74:90] = np.arange(1, 17) * 0.06 + 0.006
PITY_5STAR[90] = 1
# 绝区零普通4星保底概率表 基础概率9.4% 角色池其中角色为7.05% 音擎为2.35% 十抽保底 综合14.4%
# 角色池不触发UP机制时角色占比 1/2 音擎占比 1/2
# 角色池触发UP机制时UP角色占比 1/2 其他角色占比 1/4 音擎占比 1/4
PITY_4STAR = np.zeros(11)
PITY_4STAR[1:10] = 0.094
PITY_4STAR[10] = 1

# 绝区零音擎5星保底概率表 基础概率1% 综合概率2% 80保底 75%概率单UP
PITY_W5STAR = np.zeros(81)
PITY_W5STAR[1:65] = 0.01
PITY_W5STAR[65:80] = np.arange(1, 16) * 0.06 + 0.01
PITY_W5STAR[80] = 1
# 绝区零音擎4星保底概率表 基础概率15% 其中音擎占13.125% 角色占1.875% 10抽保底 综合概率18% 75%概率UP
# 音擎池不触发UP机制时音擎占比 1/2 角色占比 1/2
# 音擎池触发UP机制时UP音擎占比 3/4 其他音擎占比 1/8 角色占比 1/8
PITY_W4STAR = np.zeros(11)
PITY_W4STAR[1:10] = 0.15
PITY_W4STAR[10] = 1

# 定义获取星级物品的模型
common_5star = PityModel(PITY_5STAR)
common_4star = PityModel(PITY_4STAR)
# 定义绝区零角色池模型
up_5star_character = DualPityModel(PITY_5STAR, [0, 0.5, 1])
pick_up_5star_character = ExclusiveRescreeningModel(common_5star, up_5star_character)
up_4star_character = DualPityModel(PITY_4STAR, [0, 0.5, 1])
up_4star_specific_character = DualPityBernoulliModel(PITY_4STAR, [0, 0.5, 1], 1/2)
# 定义绝区零武器池模型
common_5star_weapon = PityModel(PITY_W5STAR)
common_4star_weapon = PityModel(PITY_W4STAR)
up_5star_weapon = DualPityModel(PITY_W5STAR, [0, 0.75, 1])
pick_up_5star_weapon = ExclusiveRescreeningModel(common_5star_weapon, up_5star_weapon)
up_4star_weapon = DualPityModel(PITY_W4STAR, [0, 0.75, 1])
up_4star_specific_weapon = DualPityBernoulliModel(PITY_W4STAR, [0, 0.75, 1], 1/2)
# 定义绝区零邦布池模型
bangboo_5star = PityModel(PITY_5STAR)
bangboo_4star = PityModel(PITY_4STAR)

if __name__ == '__main__':
    print(common_5star(1).exp)
    print(common_5star_weapon(1).exp)
    print(common_4star(1).exp)
    print(common_4star_weapon(1).exp)
    print(up_5star_character(1).exp)
    print(up_5star_weapon(1).exp)
    ans = pick_up_5star_character(3, first_ten_discount=False, multi_dist=True)
    for dist in ans:
        print(dist.exp)
    # print()  # .quantile_point([0.1, 0.25, 0.5, 0.75, 0.99]))
    # print(pick_up_5star_character(1, first_ten_discount=True).quantile_point([0.1, 0.25, 0.5, 0.75, 0.99]))
    print(f"一直单抽获得第一个自选UP的消耗为{pick_up_5star_character(2).exp}")
    '''
    close_dis = 1
    pity_begin = 0
    p_raise = 0
    for i in range(60, 75+1):
        # 枚举开始上升位置
        PITY_5STAR = np.zeros(81)
        PITY_5STAR[1:i] = 0.01
        for j in range(5, 10):
            # 枚举每抽上升概率
            p_step = j / 100
            PITY_5STAR[i:80] = np.arange(1, 80-i+1) * p_step + 0.01
            PITY_5STAR[80] = 1
            common_5star = PityModel(PITY_5STAR)
            p = 1 / common_5star(1).exp
            if p > 0.02:
                # 达到要求进行记录
                if p-0.02 < close_dis:
                    close_dis = p-0.02
                    pity_begin = i
                    p_raise = p_step
                    print(p, i, p_step, PITY_5STAR[70:81])
    '''
    