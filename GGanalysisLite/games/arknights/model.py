from GGanalysisLite.distribution_1d import *
from GGanalysisLite.gacha_layers import *
from GGanalysisLite.basic_models import *

'''
    注意，本模块按公示概率进行建模，但忽略了一些情况
        如不纳入对300保底的考虑，获取1个物品的分布不会在第300抽处截断
        这么做的原因是，模型只支持一抽最多获取1个物品，若在第300抽处
        刚好抽到物品，等于一抽抽到两个物品，无法处理。对于这个问题，建议
        结合抽数再加一步分析，进行一次后处理
'''

# 设置六星概率递增表
pity_6star = np.zeros(100)
pity_6star[1:51] = 0.02
pity_6star[51:99] = np.arange(1, 49) * 0.02 + 0.02
pity_6star[99] = 1

# 获取普通六星
Common_6star = pity_model(pity_6star)
# 获取单UP六星
Single_UP_6star = pity_bernoulli_model(pity_6star, 1/2)
# 获取双UP六星中的特定六星
Dual_UP_specific_6star = pity_bernoulli_model(pity_6star, 1/4)    
# 获取限定UP六星中的限定六星
LimitedUP_6star = pity_bernoulli_model(pity_6star, 0.35)

# 获取普通五星
Common_5star = bernoulli_gacha_model(0.08)
# 获取单UP五星
Single_UP_specific_5star = bernoulli_gacha_model(0.08/2)
# 获取双UP五星中的特定五星
Dual_UP_specific_5star = bernoulli_gacha_model(0.08/2/2)
# 获取三UP五星中的特定五星
Triple_UP_specific_5star = bernoulli_gacha_model(0.08/2/3)
