from GGanalysis.games.zenless_zone_zero.gacha_model import *
from GGanalysis.markov_method import calc_stationary_distribution
from GGanalysis import calc_expectation
import numpy as np

# 对一些特别情况的期望计算
# 如复刻池采用首次十连的方法抽卡下的循环期望
# 定义状态0-10为获得UP五星数量，0-10为五星垫抽数量，0-1为大保底情况
def calc_pickup_pool_cycle_exp(gacha_model: ExclusiveRescreeningModel, base_p=0.006, eps = 1e-14):
    def get_number(num, pity, up_pity):
        return (num * 11 + pity) * 2 + up_pity
    def get_state(number):
        up_pity = number % 2
        number = number // 2
        pity = number % 11
        num  = number // 11
        return num, pity, up_pity
    # 状态最大为 (10*11+10)*2+1=241
    M = np.zeros((242, 242), dtype=float)
    # 构造转移矩阵 设定10为吸收态
    for num in range(10+1):
        if num == 0:
            # 首个UP必中
            for pity in range(10+1):
                # 没有大保底
                now_num = get_number(0, pity, 0)
                next_num = get_number(1, 0, 0)
                M[next_num, now_num] = base_p
                next_num = get_number(0, min(10, pity+1), 0)
                M[next_num, now_num] += 1-base_p
                # 有大保底的情况
                now_num = get_number(0, pity, 1)
                next_num = get_number(1, 0, 1)
                M[next_num, now_num] = base_p
                next_num = get_number(0, min(10, pity+1), 1)
                M[next_num, now_num] += 1-base_p
        else:
            for pity in range(10+1):
                # 没有大保底
                now_num = get_number(num, pity, 0)
                # 出了五星
                # 没有中UP
                next_num = get_number(num, 0, 1)
                M[next_num, now_num] = base_p/2
                # 中了UP
                next_num = get_number(min(10, num+1), 0, 0)
                M[next_num, now_num] += base_p/2
                # 没出五星
                next_num = get_number(num, min(10, pity+1), 0)
                M[next_num, now_num] += 1-base_p

                # 有大保底
                now_num = get_number(num, pity, 1)
                # 出了五星
                next_num = get_number(min(10, num+1), 0, 0)
                M[next_num, now_num] += base_p
                # 没出五星
                next_num = get_number(num, min(10, pity+1), 1)
                M[next_num, now_num] += 1-base_p
    # 抽10次后，如何没出则继续抽到出。如果出了则停止，保留当前剩余垫抽数。
    # 研究永续的情况，需要将没出UP的剩余抽数映射回到0，出了UP的剩余抽数映射回到出UP数量为0的位置
    # 定义切换卡池矩阵
    M_S = np.zeros((242, 242), dtype=float)
    for num in range(10+1):
        if num == 0:
            # 没有获得至少1个的都清零
            for pity in range(10+1):
                # 无大保底
                now_num = get_number(0, pity, 0)
                next_num = get_number(0, 0, 0)
                M_S[next_num, now_num] = 1
                # 有大保底
                now_num = get_number(0, pity, 1)
                next_num = get_number(0, 0, 0)
                M_S[next_num, now_num] = 1
        else:
            for pity in range(10+1):
                # 无大保底
                now_num = get_number(num, pity, 0)
                next_num = get_number(0, pity, 0)
                M_S[next_num, now_num] = 1
                # 有大保底
                now_num = get_number(num, pity, 1)
                next_num = get_number(0, pity, 1)
                M_S[next_num, now_num] = 1
    M_10 = np.linalg.matrix_power(M, 10)
    M_POOL = M_S @ M_10
    init_state = np.zeros(242, dtype=float)
    init_state[get_number(0, 0, 0)] = 1
    # 获得平稳后的每轮卡池开始时保底状态，截断超低概率部分
    pity_dist = calc_stationary_distribution(M_POOL)
    pity_dist[pity_dist < eps] = 0.0
    # 获得平稳后抽10抽的分布，用于提取当前状态下获得N个道具的概率
    dist_10 = M_10 @ pity_dist

    # 计算前10抽部分出的道具数量期望
    def get_num_mask(num):
        mask = np.zeros(242, dtype=bool)
        mask[get_number(num, np.arange(11), 0)] = True
        mask[get_number(num, np.arange(11), 1)] = True
        return mask
    item_num_dist = np.zeros(11)
    for i in range(11):
        item_num_dist[i] = sum(dist_10 * get_num_mask(i))
    get_item_exp_10 = calc_expectation(item_num_dist)  
    # print(item_num_dist, get_item_exp_10)
    # 对于没出的垫抽部分计算期望
    use_pull_exp = 0
    for i in range(0, 10+1):
        # 小保底
        use_pull_exp += dist_10[get_number(0, i, 0)] * gacha_model(1, item_pity=i, up_pity=0).exp
        # 大保底对于第一个道具也没有作用，但是照样填写
        use_pull_exp += dist_10[get_number(0, i, 1)] * gacha_model(1, item_pity=i, up_pity=1).exp

    total_pull_exp = use_pull_exp + 10  # 无论如何都会用十抽，所以即使情况不同统一加十就行
    total_item_exp = get_item_exp_10 + sum(dist_10 * get_num_mask(0))  # 前十抽出的部分+没出的部分一定为1
    return total_pull_exp, total_item_exp
    # for i in range(0, 10+1):
    #     print(f'垫{i}抽小保底概率为{pity_dist[get_number(0, i, 0)]}')
    #     print(f'垫{i}抽大保底概率为{pity_dist[get_number(0, i, 1)]}')
    # print(f'{sum(pity_dist):.6f}')  # pity_dist

if __name__ =='__main__':
    # 不考虑十连八折的情况下
    print(calc_pickup_pool_cycle_exp(pick_up_5star_character, base_p=0.006))
    print(calc_pickup_pool_cycle_exp(pick_up_5star_weapon, base_p=0.01))
    pass