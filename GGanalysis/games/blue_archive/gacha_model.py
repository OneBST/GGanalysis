from GGanalysis.basic_models import cdf2dist
from GGanalysis.markov.coupon_collection import GeneralCouponCollection
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
    'pull_exchange_dp_1',
    'pull_exchange_dp_10',
    'get_common_refund',
    'STANDER_3STAR',
    'EXCHANGE_PULL',
]

P_3 = 0.03
P_2 = 0.185
P_1 = 0.785

"""
十连抽时2星综合概率
= E(十连抽到的2星数量) / 10
= (E(前9抽抽到的2星数量) + P(第10抽抽到2星)) / 10
= (9 * P(第1抽抽到2星) + P(第10抽抽到2星 | 前9抽未抽到2星及以上) * P(前9抽未抽到2星及以上) + P(第10抽抽到2星 | 前9抽已抽到2星及以上) * P(前9抽已抽到2星及以上)) / 10
= (9 * 18.50 % + 97 % * 78.50 % ^ 9 + 18.50 % * (1 - 78.50 % ^ 9)) / 10
= 198539059901039401398249/1024000000000000000000000
= 0.19388580068460878
"""
P_2_TENPULL = (9 * P_2 + (1 - P_3) * P_1 ** 9 + P_2 * (1 - P_1 ** 9)) / 10
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
        
def pull_exchange_dp_1(base_p=P_3, up_p=P_3UP, other_charactors=STANDER_3STAR, exchange_pos=EXCHANGE_PULL):
    '''在重复角色获得神名文字无价值情情况下用最低抽数集齐的策略，即会切换卡池抽，每次单抽'''
    # 即获得了A后就换到B池继续抽的情况
    import numpy as np
    # A为默认先抽角色 B默认为后抽角色
    # 设置状态有三：两者都集齐了的状态11，获得了A的状态01，获得了B的状态10，两者都没有的00
    max_pull = exchange_pos * 2
    M = np.zeros((max_pull+1, 4), dtype=float)
    M[0, 0] = 1
    # 根据状态转移进行DP
    for pull in range(1, max_pull+1):
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

    cdf = M[:, 3].copy()
    # 第一井后，已经抽到任一目标时可兑换另一个目标。
    cdf[exchange_pos:] += M[exchange_pos:, 1]
    cdf[exchange_pos:] += M[exchange_pos:, 2]
    # 第二井保证可以用两次兑换集齐两个目标。
    cdf[max_pull] = 1
    return cdf

def pull_exchange_dp_10(base_p=P_3, up_p=P_3UP, other_charactors=STANDER_3STAR, exchange_pos=EXCHANGE_PULL):
    '''在重复角色获得神名文字无价值情情况下用最低抽数集齐的策略，即会切换卡池抽，每次十连抽'''
    a, b = up_p, (base_p-up_p)/other_charactors
    # 设置每种状态下的转移矩阵，开始位置为先抽A卡池
    M_A = np.array([[1-a-b, 0, 0, 0],
                    [a, 1-b, 0, 0],
                    [b, 0, 1-a, 0],
                    [0, b, a, 1]])
    M_B = np.array([[1-a-b, 0, 0, 0],
                    [b, 1-a, 0, 0],
                    [a, 0, 1-b, 0],
                    [0, a, b, 1]])
    # 十连自乘10次
    M_A_10 = np.linalg.matrix_power(M_A, 10)
    M_B_10 = np.linalg.matrix_power(M_B, 10)
    # 得到策略组合后的矩阵
    M = np.hstack((M_A_10[:, [0]], M_B_10[:, [1]], M_A_10[:, [2]], M_B_10[:, [3]]))
    
    # 递推得到结果
    k = int(exchange_pos/10) * 2 # 十连次数
    ans = np.zeros((k+1, 4), dtype=float)
    x = np.array([1, 0, 0, 0])
    # 计算所有情况
    for i in range(0, k+1):
        ans[i, :] = x
        x = np.matmul(M, x)

    # 对井的情况进行处理
    ans[20:, 3] += ans[20:, 1]
    ans[20:, 3] += ans[20:, 2]
    ans[40:, 3] = 1

    return ans[:, 3]

def no_exchange_dp_10(base_p=P_3, up_p=P_3UP, other_charactors=STANDER_3STAR, exchange_pos=EXCHANGE_PULL):
    '''上面的函数的不切换抽到井版本'''
    a, b = up_p, (base_p-up_p)/other_charactors
    # 设置每种状态下的转移矩阵，开始位置为先抽A卡池
    M_A = np.array([[1-a-b, 0, 0, 0],
                    [a, 1-b, 0, 0],
                    [b, 0, 1-a, 0],
                    [0, b, a, 1]])
    # 十连自乘10次
    M_A_10 = np.linalg.matrix_power(M_A, 10)
    # 得到策略组合后的矩阵
    M = M_A_10
    
    # 递推得到结果
    k = int(exchange_pos/10) * 2 # 十连次数
    ans = np.zeros((k+1, 4), dtype=float)
    x = np.array([1, 0, 0, 0])
    # 计算所有情况
    for i in range(0, k+1):
        ans[i, :] = x
        x = np.matmul(M, x)

    # 对井的情况进行处理
    ans[20:, 3] += ans[20:, 1]
    ans[20:, 3] += ans[20:, 2]
    ans[40:, 3] = 1

    return ans[:, 3]

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
    import matplotlib.pyplot as plt
    from GGanalysis.distribution_1d import *
    
    # 平稳时来自保底的比例
    print("平稳时来自保底的比例为", 1/(EXCHANGE_PULL*0.03+1), "UP来自保底的比例为", 1/(EXCHANGE_PULL*0.007+1))

    # 十连的实际综合概率
    print(f"十连时三星概率{P_3} 两星概率{P_2_TENPULL} 一星概率{P_1}，两星概率相比不十连提升{100*(P_2_TENPULL/P_2-1)}%")

    # 计算单抽仅获取UP抽数期望（拥有即算，在200抽抽到时虽然可以多井一个但也算保证获取一个UP
    model = SimpleDualCollection(other_charactors=STANDER_3STAR)
    both_ratio, a_ratio, b_ratio, none_ratio = model.get_dist(calc_pull=EXCHANGE_PULL*2)
    temp_dist_1 = both_ratio+a_ratio
    temp_dist_1 = temp_dist_1[:201]
    temp_dist_1[200] = 1
    temp_dist_1 = cdf2dist(temp_dist_1)
    print("含井单抽仅获取UP角色的抽数期望", temp_dist_1.exp)
    # 含井单抽仅获取UP角色的抽数期望 107.80200663752993

    # 计算十连抽仅获取UP抽数期望（拥有即算，在200抽抽到时虽然可以多井一个但也算保证获取一个UP
    temp_dist_1_tenpull = both_ratio+a_ratio
    temp_dist_1_tenpull = temp_dist_1_tenpull[:201]
    temp_dist_1_tenpull[200] = 1
    temp_dist_1_tenpull = cdf2dist(temp_dist_1_tenpull)
    temp_dist_1_tenpull = dist_squeeze(temp_dist_1_tenpull, 10)
    print("含井十连仅获取UP角色的抽数期望", temp_dist_1_tenpull.exp*10)
    # 含井十连仅获取UP角色的抽数期望 111.24149841754267

    # 计算获取同期两个UP，采用抽1井1方法的期望（按照每次十连，出了对应学生就换池的方法进行)
    temp_dist_dual_1 = cdf2dist(pull_exchange_dp_1())
    print("一直单抽，抽1井1获得角色即换池策略下获得同时UP的两类角色的抽数期望", temp_dist_dual_1.exp)
    # 一直单抽，抽1井1获得角色即换池策略下获得同时UP的两类角色的抽数期望 181.71405074051484

    # 计算获取同期两个UP，采用抽1井1方法的期望（按照每次十连，出了对应学生就换池的方法进行)
    temp_dist_dual_10 = cdf2dist(pull_exchange_dp_10())
    print("一直十连抽，抽1井1获得角色即换池策略下获得同时UP的两类角色的抽数期望", temp_dist_dual_10.exp*10)
    # 一直十连抽，抽1井1获得角色即换池策略下获得同时UP的两类角色的抽数期望 185.85079879472937

    # 计算获取同期两个UP，采用抽1井1方法的期望（按照每次十连，但是不换池的方法)
    temp_dist_dual = cdf2dist(no_exchange_dp_10())
    print("不换池情况的期望", temp_dist_1.exp*10)

    # 计算含/不含井的神名文字返还

    # 粗估UP池内抽满一个角色的期望(含兑换，按无限平均估算，按5:1兑换并认为其他角色都已拥有)，3星升级5星需要220神名文字
    # 这个计算严重失真，因为不是这么有机会可以井到
    E_up_pull = (P_3UP + 1/200) * 100 # + get_common_refund(rc3=1, has_up=1) * 0.2
    left = 220 # - 107.80200663752993 * get_common_refund(rc3=0.5, has_up=0) * 0.2
    print("拥有角色后，每抽平均获得角色神名文字:", E_up_pull)
    print("抽满一个角色的估计抽数是:", 107.80200663752993 + left / E_up_pull)
    # 抽满一个角色的估计抽数是: 291.13533997086324

    # 新卡池逻辑
    ans = np.zeros(201)
    left = 1
    for i in range(1, 100):
        ans[i] = left * P_3UP
        left *= 1 - P_3UP
    ans[100] = 0.5 * left
    left *= 0.5
    for i in range(101, 200):
        ans[i] = left * P_3UP
        left *= 1 - P_3UP
    ans[200] = left

    # 抽1个
    new_1_up3 = FiniteDist(ans)
    # 抽2个
    new_2_up3 = new_1_up3 ** 2

    # 对比新旧卡池获取1个、2个UP角色的累计分布
    old_1_up3 = temp_dist_1
    old_2_up3 = cdf2dist(pull_exchange_dp_1())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    comparison_data = (
        (axes[0], old_1_up3, new_1_up3, 'Obtain 1 UP'),
        (axes[1], old_2_up3, new_2_up3, 'Obtain 2 UPs'),
    )
    for ax, old_dist, new_dist, title in comparison_data:
        ax.plot(
            np.arange(len(old_dist)),
            old_dist.cdf,
            label=f'Old (E={old_dist.exp:.2f})',
            drawstyle='steps-post',
            linewidth=2,
        )
        ax.plot(
            np.arange(len(new_dist)),
            new_dist.cdf,
            label=f'New (E={new_dist.exp:.2f})',
            drawstyle='steps-post',
            linewidth=2,
        )
        ax.set_title(title)
        ax.set_xlabel('Pulls')
        ax.set_ylabel('CDF')
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle('Blue Archive: Old vs New Banner')
    fig.tight_layout()

    # 对比新旧卡池获取1个、2个UP角色的概率质量分布
    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    distribution_data = (
        (axes_dist[0], old_1_up3, new_1_up3, 'Obtain 1 UP'),
        (axes_dist[1], old_2_up3, new_2_up3, 'Obtain 2 UPs'),
    )
    for ax, old_dist, new_dist, title in distribution_data:
        ax.plot(
            np.arange(len(old_dist)),
            old_dist.dist,
            label='Old',
            drawstyle='steps-mid',
            linewidth=1.5,
        )
        ax.plot(
            np.arange(len(new_dist)),
            new_dist.dist,
            label='New',
            drawstyle='steps-mid',
            linewidth=1.5,
        )
        ax.set_title(title)
        ax.set_xlabel('Pulls')
        ax.set_ylabel('Probability')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()

    fig_dist.suptitle('Blue Archive: Old vs New Banner Distribution')
    fig_dist.tight_layout()
    plt.show()
