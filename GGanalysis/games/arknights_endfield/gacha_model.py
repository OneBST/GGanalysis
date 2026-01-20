from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *
from GGanalysis.games.arknights_endfield.reward_calculation import RewardRule, reward_positions, apply_interval_reward, apply_item_reward_with_distribution

'''
    注意，本模块按明日方舟：终末地公示机制进行简单建模，非统计版本
'''

__all__ = [
    'PITY_6STAR',
    'PITY_5STAR',
    'HardGuarantee_UP6star',
    'IntervalAutoReward_UP6star',
    'PITY_W6STAR',
    'PITY_W5STAR',
    'HardGuarantee_UPW6star',
    'IntervalAutoReward_UPW6star',
    'common_6star',
    'up_6star_first_character',
    'up_6star_character_after_first',
    'up_6star_character_reward',
    'weapon_6star',
    'up_6star_first_weapon',
    'up_6star_weapon_after_first',
    'up_6star_weapon_reward',
    'common_5star',
    'weapon_5star',
]

class AKESinglePityModel(GachaModel):
    '''终末地硬保底模型'''
    def __init__(self, pity_p, up_p, up_pity_pull):
        super().__init__()
        self.pity_p = pity_p
        self.up_p = up_p
        self.up_pity_pull = up_pity_pull
        self.common_up_model = PityBernoulliModel(pity_p, up_p)
    
    def __call__(self, item_num: int=1, multi_dist: bool=False, item_pity=0, single_up_pity=0) -> Union[FiniteDist, list]:
        '''
        抽取个数 是否要返回多个分布列 道具保底进度 单次赠送保底进度
        '''
        if single_up_pity == -1:
            # 已经不是第一个UP了，模型退化为原始形态
            return self.common_up_model(item_num=item_num, multi_dist=multi_dist, item_pity=item_pity)
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
        # 处理一次性保底位置
        first_dist = conditional_dist[:self.up_pity_pull-single_up_pity+1]
        first_dist[self.up_pity_pull-single_up_pity] = 1-sum(first_dist[:self.up_pity_pull-single_up_pity])
        first_dist = FiniteDist(first_dist)  # 定义抽第一个道具的分布
        if item_num == 1:
            return first_dist
        stander_dist = self.common_up_model(1)
        return first_dist * stander_dist ** (item_num-1)

class AKERewardModel(GachaModel):
    def __init__(self, base_model, reward_rule):
        super().__init__()
        self.base_model = base_model
        self.reward_rule = reward_rule
    
    def __call__(self, item_num: int=1, multi_dist: bool=False, item_pity=0, single_up_pity=0, reward_counter=0) -> Union[FiniteDist, list]:
        if item_num == 0:
            return FiniteDist([1])
        # 计算抽到要求个数所需的cdf
        dist_list = self.base_model(item_num, multi_dist=True, item_pity=item_pity, single_up_pity=single_up_pity)
        raw_cdf_list = [dist.cdf for dist in dist_list]
        # 获得应用了固定抽数额外返还的cdf
        local_reward_rule = lambda x: self.reward_rule(x) - reward_counter  # 考虑垫了reward个数后的情况
        reward_cdf_list = apply_interval_reward(raw_cdf_list, reward_rule=local_reward_rule)
        # 处理为分布列
        reward_dist_list = [cdf2dist(cdf) for cdf in reward_cdf_list]
        for dist, cdf in zip(reward_dist_list, reward_cdf_list):
            dist.cdf = cdf
        # 如果 multi_dist 参数为真，返回抽取 [1, 抽取个数] 个道具的分布列表
        if multi_dist:
            return reward_dist_list
        # 其他情况正常返回
        return reward_dist_list[item_num]

class AKECharacterRewardModel(GachaModel):
    def __init__(self, base_model, reward_rule, get_extra_pull_pos=30, get_extra_pull=10, base_p=0.004):
        super().__init__()
        self.base_model = base_model
        self.reward_rule = reward_rule
        self.get_extra_pull_pos = get_extra_pull_pos
        self.get_extra_pull = get_extra_pull
        self.num_p = binom.pmf(np.arange(0, get_extra_pull+1), get_extra_pull, base_p)

    def __call__(self, item_num: int=1, multi_dist: bool=False, item_pity=0, single_up_pity=0, reward_counter=0, pull_reward_counter=0) -> Union[FiniteDist, list]:
        if item_num == 0:
            return FiniteDist([1])
        # 计算抽到要求个数所需的cdf
        dist_list = self.base_model(item_num, multi_dist=True, item_pity=item_pity, single_up_pity=single_up_pity)
        raw_cdf_list = [dist.cdf for dist in dist_list]
        # 获得应用了固定抽数额外返还的cdf
        local_reward_rule = lambda x: self.reward_rule(x) - reward_counter  # 考虑垫了reward个数后的情况
        reward_cdf_list = apply_interval_reward(raw_cdf_list, reward_rule=local_reward_rule)
        # 获得应用了固定位置返还抽数的cdf
        if pull_reward_counter != -1:
            pull_cdf_list = apply_item_reward_with_distribution(reward_cdf_list, self.num_p, self.get_extra_pull_pos-pull_reward_counter)
        else:
            pull_cdf_list = reward_cdf_list
        # 处理为分布列
        final_dist_list = [cdf2dist(cdf) for cdf in pull_cdf_list]
        for dist, cdf in zip(final_dist_list, pull_cdf_list):
            dist.cdf = cdf
        # 如果 multi_dist 参数为真，返回抽取 [1, 抽取个数] 个道具的分布列表
        if multi_dist:
            return final_dist_list
        # 其他情况正常返回
        return final_dist_list[item_num]


# 设置6星概率递增表
PITY_6STAR = np.zeros(81)
PITY_6STAR[1:65+1] = 0.008
PITY_6STAR[66:81] = np.arange(1, 15+1) * 0.05 + 0.008
PITY_6STAR[80] = 1
# 设置UP角色卡池UP6星保底和赠送参数
HardGuarantee_UP6star = 120  # 每个卡池的一次性保底
IntervalAutoReward_UP6star = lambda j: 240 * j  # 每240抽赠送的角色
# 角色池抽到60抽可以获得仅限下个卡池可用的免费10抽
# 设置5星概率递增表
PITY_5STAR = np.zeros(11)
PITY_5STAR[1:9+1] = 0.08
PITY_5STAR[10] = 1
# 设置武器池6星概率递增表
PITY_W6STAR = np.zeros(41)
PITY_W6STAR[1:39+1] = 0.04
PITY_W6STAR[40] = 1
# 设置UP武器卡池UP6星保底和赠送参数
HardGuarantee_UPW6star = 80  # 每个卡池的一次性保底
IntervalAutoReward_UPW6star = lambda j: 160 * j + 20  # 武器池满赠
# UP武器卡池额外物品获得规则比较复杂，第100次给一个本卡池内非UP武器自选，第180次给一个本卡池UP武器，随后每80抽给一个，UP和非UP轮换
# 设置武器池5星概率递增表
PITY_W5STAR = np.zeros(11)
PITY_W5STAR[1:9+1] = 0.15
PITY_W5STAR[10] = 1 

# 6★
common_6star = PityModel(PITY_6STAR)
up_6star_first_character = AKESinglePityModel(PITY_6STAR, 0.5, HardGuarantee_UP6star)  # single_up_pity 填写-1表示已经没有第一个UP6星的120保底
up_6star_character_after_first = PityBernoulliModel(PITY_6STAR, 0.5)  # 不考虑第一个
up_6star_character_reward_old = AKERewardModel(up_6star_first_character, IntervalAutoReward_UP6star)
up_6star_character_reward = AKECharacterRewardModel(up_6star_first_character, IntervalAutoReward_UP6star, 30, 10, 0.004)

weapon_6star = PityModel(PITY_W6STAR)
up_6star_first_weapon = AKESinglePityModel(PITY_W6STAR, 0.25, HardGuarantee_UPW6star)
up_6star_weapon_after_first = PityBernoulliModel(PITY_W6STAR, 0.25)  # 不考虑第一个
up_6star_weapon_reward = AKERewardModel(up_6star_first_weapon, IntervalAutoReward_UPW6star)

# 5★
# 5星公示基础概率为8%，更多信息还有待发掘，目前看来获取6星会重置5星保底
common_5star = PityModel(PITY_5STAR)  # 不考虑被6星重置的简单模型
weapon_5star = PityModel(PITY_W5STAR)  # 不考虑被6星重置的简单模型
up_5star_character = PityBernoulliModel(PITY_5STAR, 0.5)  # 不考虑被6星重置的简单模型

if __name__ == '__main__':
    # a = AKECharacterRewardModel(1, 1)
    # print(up_6star_character_reward_old(2).cdf[30:41])
    # exit()
    # p_0 = (1-0.004) ** 10
    # p_1 = 1 - (1-0.004) ** 1ss0
    # p_1 = 10 * 0.004 * (1-0.004)**9
    # p_2 = 1 - p_0 - p_1
    # print(up_6star_character_reward.num_p)
    # print(np.cumsum(up_6star_character_reward.num_p))
    # print(p_0, p_1, p_2)
    # temp_cdf = up_6star_character_reward_old(1).cdf
    # temp_cdf = temp_cdf * (1-p_1) + p_1
    # temp_cdf = up_6star_character_reward_old(1).cdf * p_1 + up_6star_character_reward_old(2).cdf[:121] * p_0 + p_2
    # print(up_6star_character_reward_old(1).cdf[29:41])
    # print(temp_cdf[30:41])
    # print(sum(up_6star_character_reward_old(1)[30:41]))
    # print(up_6star_character_reward(2).cdf[30:41])
    # print(up_6star_character_reward(2).exp)
    # exit()

    print('常驻池6星期望为', common_6star(1).exp)
    character_dists = up_6star_character_reward(6, True)
    for i, dist in zip(range(1, len(character_dists)), character_dists[1:]):
        print(f'从零开始恰好获取到{i}个UP6星角色的期望为{dist.exp}抽（含满赠）平均每个{dist.exp/i}抽')
    print("-"*95)
    weapon_dists = up_6star_weapon_reward(6, True)
    for i, dist in zip(range(1, len(weapon_dists)), weapon_dists[1:]):
        print(f'从零开始恰好获取到{i}个UP6星武器的期望为{dist.exp}抽（含满赠）平均每个{dist.exp/i}抽')
    pass
