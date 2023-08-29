from GGanalysis.basic_models import cdf2dist
from GGanalysis.recursion_methods import GeneralCouponCollection
from GGanalysis.basic_models import *

__all__ = [
    'P_3',
    'P_2',
    'P_1',
    'P_2_TENPULL',
    'P_1_TENPULL',
    'P_3UP',
    'P_2UP',
    'up_3star',
    'up_2star',
    'SimpleDualCollection',
    'no_exchange_dp',
    'get_common_refund',
    'STANDER_3STAR',
    'EXCHANGE_PULL',
]

P_3 = 0.03
P_2 = 0.185
P_1 = 0.785
P_2_TENPULL = (P_2 * 9 + 1 - P_3) / 10
P_1_TENPULL = 1 - 0.03 - P_2_TENPULL

P_3UP = 0.007
P_2UP = 0.03

STANDER_3STAR = 20
EXCHANGE_PULL = 200

up_3star = BernoulliGachaModel(P_3UP)
up_2star = BernoulliGachaModel(P_2UP)

class SimpleDualCollection():
    '''考虑一直抽一个池，同时集齐两个卡池UP的概率'''
    def __init__(self, base_p=P_3, up_p=P_3UP, other_charactors=STANDER_3STAR) -> None:
        self.model = GeneralCouponCollection([up_p, (base_p-up_p)/other_charactors], ['A_UP', 'B_UP'])
    
    def get_dist(self, calc_pull=EXCHANGE_PULL):
        M = self.model.collection_dp(calc_pull)
        end_state = self.model.encode_state_number(['A_UP', 'B_UP'])
        a_state = self.model.encode_state_number(['A_UP'])
        b_state = self.model.encode_state_number(['B_UP'])
        none_state = self.model.encode_state_number([])
        both_ratio = M[end_state, :]
        a_ratio = M[a_state, :]
        b_ratio = M[b_state, :]
        none_ratio = M[none_state, :]
        return both_ratio, a_ratio, b_ratio, none_ratio
        
def no_exchange_dp(base_p=P_3, up_p=P_3UP, other_charactors=STANDER_3STAR, exchange_pos=EXCHANGE_PULL):
    '''在重复角色获得神名文字无价值情情况下用最低抽数集齐的策略，即会切换卡池抽'''
    # 即获得了A后就换到B池继续抽的情况
    import numpy as np
    # A为默认先抽角色 B默认为后抽角色
    # 设置状态有三：两者都集齐了的状态11，获得了A的状态01，获得了B的状态10，两者都没有的00
    M = np.zeros((exchange_pos+1, 4), dtype=float)
    M[0, 0] = 1
    # 根据状态转移进行DP
    for pull in range(1, exchange_pos+1):
        M[pull, 0] += M[pull-1, 0] * (1-up_p-(base_p-up_p)/other_charactors)

        M[pull, 1] += M[pull-1, 0] * up_p
        M[pull, 1] += M[pull-1, 1] * (1-up_p)

        M[pull, 2] += M[pull-1, 0] * (base_p-up_p)/other_charactors
        M[pull, 2] += M[pull-1, 2] * (1-up_p)

        M[pull, 3] += M[pull-1, 2] * up_p
        M[pull, 3] += M[pull-1, 1] * up_p
        M[pull, 3] += M[pull-1, 3]

        # 以下是用于验证的不更换卡池的策略
        # M[pull, 0] += M[pull-1, 0] * (1-up_p-(base_p-up_p)/other_charactors)

        # M[pull, 1] += M[pull-1, 0] * up_p
        # M[pull, 1] += M[pull-1, 1] * (1-(base_p-up_p)/other_charactors)

        # M[pull, 2] += M[pull-1, 0] * (base_p-up_p)/other_charactors
        # M[pull, 2] += M[pull-1, 2] * (1-up_p)

        # M[pull, 3] += M[pull-1, 2] * up_p
        # M[pull, 3] += M[pull-1, 1] * (base_p-up_p)/other_charactors
        # M[pull, 3] += M[pull-1, 3]

    return M[:, 3]

def get_common_refund(rc3=0.5, rc2=1, rc1=1,has_up=False):
    '''按每个卡池抽200计算的平均每抽神名文字返还'''
    e_up = 0  # 超出1个数量的期望
    if has_up:
        e_up = EXCHANGE_PULL * P_3UP + 1  # 实质上就是线性上移
    else:
        e_up = EXCHANGE_PULL * P_3UP  # 井和超出1个部分减一相互抵消
    rc_up = e_up / EXCHANGE_PULL
    return ((P_3-P_3UP) * rc3 + rc_up) * 50  + P_2_TENPULL * 10 * rc2 + P_1_TENPULL * 1 * rc1


if __name__ == '__main__':
    import copy
    from GGanalysis.distribution_1d import *
    
    # 平稳时来自保底的比例
    print("平稳时来自保底的比例为", 1/(EXCHANGE_PULL*0.03+1), "UP来自保底的比例为", 1/(EXCHANGE_PULL*0.007+1))

    # 十连的实际综合概率
    print(f"十连时三星概率{P_3} 两星概率{P_2_TENPULL} 一星概率{P_1}，两星概率相比不十连提升{100*(P_2_TENPULL/P_2-1)}%")

    # 计算仅获取UP抽数期望（拥有即算，在200抽抽到时虽然可以多井一个但也算保证获取一个UP
    model = SimpleDualCollection(other_charactors=STANDER_3STAR)
    both_ratio, a_ratio, b_ratio, none_ratio = model.get_dist()
    temp_dist = both_ratio+a_ratio
    temp_dist = temp_dist[:201]
    temp_dist[200] = 1
    temp_dist = cdf2dist(temp_dist)
    print("含井仅获取UP角色的抽数期望", calc_expectation(temp_dist))
    # 含井仅获取UP角色的抽数期望 107.80200663752993

    # 计算获取同期两个UP，采用抽1井1方法的期望（因为大部分玩家为了200井都会抽到200，所以此时换池子抽的情况虽然理论上为集齐角色消耗抽数更少更优，但未必符合实际，于是按照一个池里一直抽计算）
    both_ratio, a_ratio, b_ratio, none_ratio = model.get_dist(calc_pull=400)
    temp_dist = copy.deepcopy(both_ratio)
    temp_dist[200:] += a_ratio[200:] + b_ratio[200:]
    temp_dist[400] = 1
    temp_dist = cdf2dist(temp_dist)
    print("含井同池内一直抽获得同时UP的两类角色的抽数期望", calc_expectation(temp_dist))
    # 含井同池内一直抽获得同时UP的两类角色的抽数期望 206.9735880064448

    # 计算无穷时间平均意义并不大，因为井过期清零

    # 计算含/不含井的神名文字返还

    # 粗估UP池内抽满一个角色的期望(含兑换，按无限平均估算，按5:1兑换并认为其他角色都已拥有)，3星升级5星需要220神名文字
    # 这个计算严重失真，因为不是这么有机会可以井到
    E_up_pull = (P_3UP + 1/200) * 100 # + get_common_refund(rc3=1, has_up=1) * 0.2
    left = 220 # - 107.80200663752993 * get_common_refund(rc3=0.5, has_up=0) * 0.2
    print("拥有角色后，每抽平均获得角色神名文字:", E_up_pull)
    print("抽满一个角色的估计抽数是:", 107.80200663752993 + left / E_up_pull)
    # 抽满一个角色的估计抽数是: 291.13533997086324

