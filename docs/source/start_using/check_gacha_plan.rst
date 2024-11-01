检查抽卡计划可行性
========================

在现实中，玩家会因为游戏中道具会分时间段开放抽取，不同时间段抽卡资源投放也不同，经常需要评估一个抽卡规划的可行性。
我们可以用在限制条件下能有多大概率达成目标来描述计划的可行性。

.. admonition:: 抽卡问题例子
    :class: note

    以绝区零为例，假设1.0版本可以获得200抽，这时候计划在1.0版本下半抽一个限定S级角色加一个限定S级武器，
    1.1版本上半可获得74抽，计划在抽一个限定S级角色，1.1版本下半可获得40抽，计划此时再抽一个限定S级角色加一个限定S级武器，
    那么按照计划把这些武器和角色都抽出来的概率有多大？

在这个问题中要计算多大概率可以达到目标，本质上是计算能在各个阶段的抽数限制下都能达成目标的玩家占比，
只需要 **在每个阶段中剔除超出抽数限制部分玩家对应概率空间** 即可。
以下代码给出了计算刚才例子中抽卡计划达成概率的可能性及达成计划的玩家中抽数花费的分布。

.. code:: Python
        
    from GGanalysis import FiniteDist
    import GGanalysis.games.zenless_zone_zero as ZZZ
    # 设定要抽的角色数量/武器数量以及阶段预算
    tasks = [
        [1, 1, 200],
        [1, 0, 74],
        [1, 1, 40],
    ]
    total_c = 0
    total_w = 0
    total_pulls = 0
    ans_dist = FiniteDist([1])
    for (num_c, num_w, task_pulls) in tasks:
        total_c += num_c
        total_w += num_w
        total_pulls += task_pulls
        ans_dist *= ZZZ.up_5star_character(num_c) * ZZZ.up_5star_weapon(num_w)
        ans_dist.dist = ans_dist.dist[:total_pulls+1]
    print("成功概率", sum(ans_dist.dist))
    ans_dist = ans_dist.normalized()  # 归一化
    print("成功玩家期望抽数消耗", ans_dist.exp)
    full_dist = ZZZ.up_5star_character(total_c)*ZZZ.up_5star_weapon(total_w)
    print("获得计划道具期望", full_dist.exp)