import math

'''
    估算免费抽数模型建立在 https://bbs.nga.cn/read.php?tid=29533283 的统计数据基础上
'''

# 计算免费获得抽数，做了一点平滑处理
def get_free_pulls(days):
    # 这里默认计算0.928的discount
    discount = 0.928
    return (8+20+days*1.76235 + 260*(1-math.exp(days*-math.log(2)/30)))/discount

# 获取购买部分抽数的价格
def get_gacha_price(pulls, days):
    price = 0
    discount = 0.865

    # 小月卡
    per_pull_price = 1.6
    buy_pulls = (days * 100/160)
    if buy_pulls >= pulls:
        price += pulls * per_pull_price
        return price
    pulls -= buy_pulls
    price += buy_pulls * per_pull_price
    # 大月卡
    per_pull_price = 8.24
    buy_pulls = (days * (680/160+5)/45)
    if buy_pulls >= pulls:
        price += pulls * per_pull_price
        return price
    pulls -= buy_pulls
    price += buy_pulls * per_pull_price
    # 双倍首充 只按重置一轮估算
    per_pull_price = 8
    buy_pulls = 163.5
    if buy_pulls >= pulls:
        price += pulls * per_pull_price
        return price
    pulls -= buy_pulls
    price += buy_pulls * per_pull_price
    # 剩下的按照648算
    price += 12.83*discount*pulls

    return price

# 获得氪金部分花费
def get_gacha_cost(pulls, days):
    free_pulls = get_free_pulls(days)
    if free_pulls > pulls:
        return 0
    pulls -= free_pulls
    return get_gacha_price(pulls, days)

# 获得买体力部分花费
def get_resin_cost(cost_per_day, days):
    return 648*(cost_per_day*days/8080)

if __name__ == '__main__':
    play_days = 365+2*30        # 游玩时间
    # print('免费获取部分', get_free_pulls(play_days))
    resin_cost_per_day = 800    # 每日体力消耗原石
    get_charactors = 74*1.5     # 总角色数量
    get_weapons = 147           # 总武器数量
    # 估算的总抽数(当然这里换成准确值算的更准)
    tot_pulls = int(get_charactors*62.297+get_weapons*53.25-(195/365)*play_days)    # 这里扣除了常驻送抽
    print('估计花销', get_gacha_cost(tot_pulls, play_days)+get_resin_cost(resin_cost_per_day, play_days))