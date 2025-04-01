from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

'''
    注意，本模块按明日方舟：终末地二测公示进行简单建模，非公测版本
    武器池由于描述不够清晰，采用猜测机制
'''

__all__ = [
    'PITY_6STAR',
    'PITY_5STAR',
    'PITY_W6STAR',
    'PITY_W5STAR',
    'common_6star',
    'weapon_6star',
    'up_6star_character',
    'common_5star',
    'weapon_5star',
]
class AKESinglePityModel(GachaModel):
    '''终末地二测硬保底模型'''
    def __init__(self, pity_p, up_p, up_pity_pull):
        super().__init__()
        self.pity_p = pity_p
        self.up_p = up_p
        self.up_pity_pull = up_pity_pull
        self.common_up_model = PityBernoulliModel(pity_p, up_p)
    
    def __call__(self, item_num: int=1, multi_dist: bool=False, item_pity=0, single_up_pity=0) -> Union[FiniteDist, list]:
        if item_num == 0:
            return FiniteDist([1])
        # 如果 multi_dist 参数为真，返回抽取 [1, 抽取个数] 个道具的分布列表
        if multi_dist:
            return self._get_multi_dist(item_num, item_pity, single_up_pity)
        # 其他情况正常返回
        return self._get_dist(item_num, item_pity, single_up_pity)
    
    # 输入 [完整分布, 条件分布] 指定抽取个数，返回抽取 [1, 抽取个数] 个道具的分布列表
    def _get_multi_dist(self, item_num: int, item_pity, single_up_pity):
        # 仿造 CommonGachaModel 里的实现
        conditional_dist = self.common_up_model(1, item_pity=item_pity)
        first_dist = conditional_dist[:self.up_pity_pull-single_up_pity+1]
        first_dist[self.up_pity_pull-single_up_pity] = 1-sum(first_dist[:self.up_pity_pull-single_up_pity])
        first_dist = FiniteDist(first_dist)  # 定义抽第一个道具的分布
        ans_list = [FiniteDist([1]), first_dist]
        if item_num > 1:
            stander_dist = self.common_up_model(1)
        for i in range(1, item_num):
            ans_list.append(ans_list[i] * stander_dist)
        return ans_list
    
    # 返回单个分布
    def _get_dist(self, item_num: int, item_pity, single_up_pity):
        conditional_dist = self.common_up_model(1, item_pity=item_pity)
        first_dist = conditional_dist[:self.up_pity_pull-single_up_pity+1]
        first_dist[self.up_pity_pull-single_up_pity] = 1-sum(first_dist[:self.up_pity_pull-single_up_pity])
        first_dist = FiniteDist(first_dist)  # 定义抽第一个道具的分布
        if item_num == 1:
            return first_dist
        stander_dist = self.common_up_model(1)
        return first_dist * stander_dist ** (item_num-1)

# 设置6星概率递增表
PITY_6STAR = np.zeros(81)
PITY_6STAR[1:65+1] = 0.008
PITY_6STAR[66:81] = np.arange(1, 15+1) * 0.05 + 0.008
PITY_6STAR[80] = 1
# 设置5星概率递增表
PITY_5STAR = np.zeros(11)
PITY_5STAR[1:9+1] = 0.08
PITY_5STAR[10] = 1
# 设置武器池6星概率递增表
PITY_W6STAR = np.zeros(41)
PITY_W6STAR[1:39+1] = 0.04
PITY_W6STAR[40] = 1
# 设置武器池5星概率递增表
PITY_W5STAR = np.zeros(11)
PITY_W5STAR[1:9+1] = 0.15
PITY_W5STAR[10] = 1

# ★★★★★★
common_6star = PityModel(PITY_6STAR)
up_6star_character_after_first = PityBernoulliModel(PITY_6STAR, 0.5)  # 不考虑第一个
up_6star_character = AKESinglePityModel(PITY_6STAR, 0.5, 120)
weapon_6star = PityModel(PITY_W6STAR)
up_weapon_6star_first = DualPityModel(PITY_W6STAR, [0, 0.25, 1])  # 每卡池的首个UP武器
up_weapon_6star_after_first = PityBernoulliModel(PITY_W6STAR, 0.25)  # 每卡池的首个UP武器
up_6star_weapon = AKESinglePityModel(PITY_W6STAR, 0.25, 80)
# ★★★★★
# 五星公示基础概率为8%，更多信息还有待发掘，目前看来获取6星会重置5星保底
common_5star = PityModel(PITY_5STAR)  # 不考虑被6星重置的简单模型
weapon_5star = PityModel(PITY_W5STAR)  # 不考虑被6星重置的简单模型
up_5star_character = PityBernoulliModel(PITY_5STAR, 0.5)  # 不考虑被6星重置的简单模型
    
if __name__ == '__main__':
    pass
