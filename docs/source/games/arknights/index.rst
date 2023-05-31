.. _arknights_gacha_model:

明日方舟抽卡模型
========================

GGanalysis 使用基本的抽卡模板模型结合 `明日方舟抽卡系统参数 <https://www.bilibili.com/read/cv20251111>`_ 定义了一系列可以直接取用的抽卡模型。需要注意的是明日方舟的抽卡系统模型的确定程度并没有很高，使用时需要注意。

此外，还针对性编写了如下模板模型：

    适用于计算 `定向选调 <https://www.bilibili.com/read/cv22596510>`_ 时获得特定UP六星干员的模型
    :class:`~GGanalysis.games.arknights.AKDirectionalModel`

    适用于计算通过 `统计数据 <https://www.bilibili.com/video/BV1ib411f7YF/>`_ 发现的类型硬保底的模型
    :class:`~GGanalysis.games.arknights.AKHardPityModel`

    适用于计算集齐多种六星的模型（不考虑300井、定向选调及类型硬保底机制）
    :class:`~GGanalysis.games.arknights.AK_Limit_Model`

.. 本节部分内容自一个资深的烧饼编写文档修改而来，MilkWind 增写内容

参数意义
------------------------

    - ``item_num`` 需求道具个数，由于 sphinx autodoc 的 `bug <https://github.com/sphinx-doc/sphinx/issues/9342>`_ 在下面没有显示

    - ``multi_dist`` 是否以列表返回获取 1-item_num 个道具的所有分布列，默认为False

    - ``item_pity`` 道具保底状态，通俗的叫法为水位、垫抽，默认为0

    - ``type_pity`` 定向选调的类型保底状态，默认为0，即该机制尚未开始触发；若为其它数，如20，那么就意味着定向选调机制已经垫了20抽，还有130抽就位于该保底机制的触发范围了（算上偏移量，实际为131抽）

    - ``calc_pull`` 采用伯努利模型时最高计算的抽数，高于此不计算（仅五星采用伯努利模型）

六星模型
------------------------

**获取任意六星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.common_6star

.. code:: python

    import GGanalysis.games.arknights as AK
    dist = AK.common_6star(item_num=1)
    print('抽到六星的期望抽数为：{}'.format(dist.exp))  # 34.59455493520977

**无定向选调获取标准寻访-单UP六星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.single_up_6star_old

**有定向选调获取标准寻访-单UP六星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.single_up_6star

.. code:: python

    import GGanalysis.games.arknights as AK
    dist = AK.single_up_6star(item_num=1, item_pity=0, type_pity=0)
    print('4.6寻访机制更新后，无水位时抽到单up六星的期望抽数为：{}'.format(dist.exp))
    
.. container:: output stream stdout

    ::

        4.6寻访机制更新后，无水位时抽到单up六星的期望抽数为：66.16056206529494

**无类型硬保底轮换池获取特定六星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.dual_up_specific_6star_old

**有类型硬保底时轮换池获取特定六星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.dual_up_specific_6star

.. code:: python

    import GGanalysis.games.arknights as AK
    dist = AK.dual_up_specific_6star(item_num=1)
    print('准备100抽，从轮换池捞出玛恩纳的概率只有：{}%'.format(sum(dist[:100+1]) * 100))

.. container:: output stream stdout

    ::

        准备100抽，从轮换池捞出玛恩纳的概率只有：49.60442859476116%

**双UP限定池获取特定六星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.limited_up_6star

    
    需要注意的是，此模型返回的结果是不考虑井的分布。如需考虑井需要自行进行一定后处理。


.. code:: python

    import GGanalysis.games.arknights as AK
    dist = AK.limited_up_6star(item_num=5)
    print('一井满潜限定的概率：{}%'.format(sum(dist_4[:300+1]) * 100))

.. container:: output stream stdout

    ::

        一井满潜限定的概率：14.881994954229667%

**双UP限定池集齐两种UP六星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.limited_both_up_6star

.. code:: python

    import GGanalysis.games.arknights as AK
    dist = AK.limited_both_up_6star()
    print('全六党吃井概率：{}%'.format((1-sum(dist[:300+1])) * 100))

.. container:: output stream stdout

    ::

        全六党吃井概率：7.130522684168872%

五星模型
------------------------

.. attention:: 

   明日方舟五星干员实际上有概率递增的星级保底机制，但其保底进度会被六星重置。这里对五星模型采用了近似，认为其是一个概率为考虑了概率递增的伯努利模型。另外，此处提供的五星模型也没有考虑类型保底。
   
   此外明日方舟五星模型没有采用 :class:`~GGanalysis.BernoulliLayer` 构建模型，而是直接采用了 :class:`~GGanalysis.BernoulliGachaModel` ，当设置 ``calc_pull`` 太低时，返回的分布概率和可能距离 1 有相当大差距，需要适当设高。

**获取任意五星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.common_5star

**获取单UP五星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.single_up_specific_5star

**获取双UP中特定五星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.dual_up_specific_5star

**获取三UP中特定五星干员**

.. automethod:: GGanalysis.games.arknights.gacha_model.triple_up_specific_5star


自定义抽卡模型例子
------------------------

.. attention::

    ``AKDirectionalModel`` 可以为某个 ``FiniteDist`` 类型的，无保底类型的分布载入定向选调机制。

    ``AKHardPityModel`` 可以为某个 ``FiniteDist`` 类型的，无保底类型的分布载入类型硬保底机制。

    ``AK_Limit_Model`` 未加入新增的定向选调机制，其使用的 `CouponCollectorLayer` 不考虑集齐多套的需求。这个模型接下来可能重写，如果想要在其他地方引用的话可以先临时复制代码出来本地使用，或是将 `AK_Limit_Model` 加入__all__公开列表进行调用。

**联合行动池集齐三种UP六星干员**

.. code:: python

    import GGanalysis.games.arknights as AK
    triple_up_specific_6star = AK.AK_Limit_Model(AK.PITY_6STAR, 1, total_item_types=3, collect_item=3)
    dist = triple_up_specific_6star(item_pity=5) # （默认）期望集齐一轮，此前垫了5抽
    print('期望抽数为：{}'.format(dist.exp)) # 期望抽数为：188.63258247595024
    print('方差为：{}'.format(dist.var)) # 方差为：10416.175324956945
    print('100抽以内达成目标的概率为：{}%'.format(sum(dist[:100+1]) * 100)) # 100抽以内达成目标的概率为：16.390307170816875%

**定向寻访池获取特定六星干员**

.. code:: python

    import GGanalysis as gg
    import GGanalysis.games.arknights as AK
    # 六星100%概率，UP三个，故抽到目标UP六星的概率为1/3
    triple_up_specific_6star = gg.PityBernoulliModel(AK.PITY_6STAR, 1 / 3) # 尚未证实定向寻访是否存在类型硬保底机制，仅使用保底伯努利模型
    dist = triple_up_specific_6star(2) # 在定向寻访池期望抽到目标六星干员两次，此前没有垫抽
    print('期望抽数为：{}'.format(dist.exp)) # 期望抽数为：207.56732961125866

**双UP限定池获取特定权值提升的非UP六星干员**

    非UP六星干员在六星中占比采用下式计算
    
    .. math:: p = 0.3\frac{5}{5 * \text{secondary up number} + \text{others number}}
    
    其中 ``secondary up number`` 为权值提升的非UP六星干员数量， ``others number`` 为除了主要UP干员和权值提升的非UP六星干员，其他准许获取的六星干员的数量。

.. code:: python

    import GGanalysis as gg
    others = 71 # 假设除了主要UP干员和权值提升的非UP六星干员外，其他准许获取的六星干员的数量为71
    triple_second_up_specific_6star = gg.PityBernoulliModel(AK.PITY_6STAR, 0.3 / (5 * 3 + others) * 5) # 在当前卡池内，权值提升的非UP六星干员数量一般为3
    success_count = 3 # 期望抽到某个权值提升的非UP六星干员三次
    dist = triple_second_up_specific_6star(success_count, True) # `multi_dist` 为True表示以列表形式返回分布
    for i in range(1, success_count + 1):
        print(f"抽到第{i}个目标干员~期望抽数：{round(dist[i].exp, 2)}，方差：{round(dist[i].var, 2)}") # 结果保留两位小数

    # 抽到第1个目标干员~期望抽数：1983.42，方差：3890458.19
    # 抽到第2个目标干员~期望抽数：3966.84，方差：7780916.38
    # 抽到第3个目标干员~期望抽数：5950.26，方差：11671374.57

    # 不太建议计算非主要UP的干员的数据，分布会很长

**标准寻访-单UP池中集齐两种UP五星干员**

.. code:: python

    import GGanalysis.games.arknights as AK
    both_up_5star = AK.AK_Limit_Model(AK.PITY_5STAR, 0.5, total_item_types=2, collect_item=2)
    dist = both_up_5star()  # 期望在轮换单UP池中抽到两个UP的五星干员，此前没有垫抽
    print('期望抽数为：{}'.format(dist.exp))  # 期望抽数为：63.03402819816313

**自定义带有类型硬保底的模型**

.. code:: python

    # 添加定向选调前已知存在类型硬保底的卡池为标准寻访中的单UP和双UP池，其它卡池暂无证据表明存在此机制，此处仅为演示如何定义此类模型，不能当做机制参考
    import GGanalysis as gg
    import GGanalysis.games.arknights as AK
    # 假设定向寻访池存在类型硬保底机制

    # 六星100%概率，UP三个，故抽到目标UP六星的概率为1/3
    triple_up_specific_6star_without_hard_pity = gg.PityBernoulliModel(AK.PITY_6STAR, 1 / 3) # 没有硬保底
    triple_up_specific_6star_has_hard_pity = AK.AKHardPityModel(triple_up_specific_6star_without_hard_pity(1), AK.p2dist(AK.PITY_6STAR), type_pity_gap=200, item_types=3, up_rate=1, type_pull_shift=1) # 载入硬保底
    dist = triple_up_specific_6star_has_hard_pity(2) # 在定向寻访池期望抽到目标六星干员两次，此前没有垫抽
    print('期望抽数为：{}'.format(dist.exp)) # 期望抽数为：207.0218117279958