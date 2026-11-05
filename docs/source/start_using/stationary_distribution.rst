平稳分布时概率的计算工具
========================

使用转移矩阵方法计算复合类型保底的概率
----------------------------------------

.. code:: Python
    
    import numpy as np
    from GGanalysis.markov import PriorityPityChain

    # 两类互斥道具：高优先级固定概率 1/2，低优先级第二抽原始概率为 1。
    # 高优先级会占用概率，但不重置低优先级进度。
    probe = PriorityPityChain([[0, 0.5], [0, 0.25, 1]],
                             extra_state=0, remove_pity=False)
    rates = probe.stationary_rates()
    assert np.allclose(rates, [0.5, 0.4])
    print(rates)

这里低优先级的实际概率受高优先级占用限制，因此即使原始表达到 1，
也不是每两抽必定获得一次。``remove_pity=True`` 则会同时改变重置规则。
游戏应用可参考各游戏目录的 ``stationary_p.py``；大型状态空间需检查求解收敛，
不能仅根据矩阵已构建就认为稳态结果可靠。

相邻两次获得同类道具的抽数间隔
------------------------------

.. code:: Python

    from GGanalysis.markov import PriorityPityChain

    probe = PriorityPityChain([[0, 0.5], [0, 0.25, 1]], extra_state=0)
    # 第 1 类道具的真实相邻获得间隔，计算到 500 抽
    interval = probe.interarrival_dist(item_type=1, max_steps=500)
    print(interval.dist[1:])
    print(interval.tail_mass)  # 间隔超过 500 抽的概率

长期平稳情况下，固定抽数内获得指定道具件数的分布
------------------------------------------------

.. code:: Python
    
    import numpy as np
    from GGanalysis.markov import stationary_item_count_distribution

    # 单类道具第一抽概率 1/2，第二抽必成，起始保底进度按长期稳态混合。
    counts = stationary_item_count_distribution([0, 0.5, 1], 2)
    assert np.allclose(counts, [0, 2 / 3, 1 / 3])

也可直接查询多类别保底链中的指定类别，保留其他类别对保底的影响：

.. code:: Python

    import numpy as np
    from GGanalysis.markov import PriorityPityChain

    probe = PriorityPityChain([[0, 0.5], [0, 0.25, 1]], extra_state=0)
    # 返回 FiniteDist。
    counts = probe.item_num_dist(item_type=1, pull=10)
    # 显式指定完整初态，不使用平稳混合。
    fresh = probe.item_num_dist(1, 2, initial=probe.space.delta([0, 0]))
    assert np.allclose(fresh.dist, [0.375, 0.5625, 0.0625])

数量分布默认从随机时刻开始；相邻获得间隔从命中后的状态开始。
这两种初态不同，不能用同一个初态代替。单类别便捷函数继续返回数组，
类内 ``item_num_dist`` 则统一返回本包的 ``FiniteDist``。


结果的适用条件
------------------------

随机观察一抽之前的稳态与刚命中目标时的状态分布不同。
``interarrival_dist`` 使用命中后的平稳起点，不是任意指定垫抽的首次等待时间；
``stationary_item_count_distribution`` 使用长期随机时刻的起点，也不是零垫抽。
间隔尾部截断时检查 ``tail_mass``，不要归一化后当作完整间隔。
求解方法及收敛条件见 :doc:`../reference_manual/markov`，
命中边界与完整状态的区别见 :doc:`../reference_manual/hit_process`。
