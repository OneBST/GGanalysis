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

术语释义：

    定向选调：该机制目前仅存在于标准寻访-单UP池。如果连续150次没有获得该限时寻访中出率上升的六星干员，则下一次招募到的六星干员必定为该限时寻访中出率上升的六星干员。该机制在当期寻访中仅生效一次。同时，当期寻访的定向选调累计次数会在该寻访结束时清零，不会累计到后续的其他【标准寻访】中。

    类型硬保底：该机制目前仅发现存在于标准寻访-双UP轮换池（不包含中坚寻访-双UP轮换池）。对于六星干员，每200抽没有获得过当期UP的，尚未获得的六星干员之一，则从下一抽开始，获得的六星干员必定为之前没有获得的当期UP的六星干员之一；对于五星干员，则是每50抽。需要注意，尚未知该机制的累计次数是否会跨卡池继承。该规律由 `一个资深的烧饼- <https://space.bilibili.com/456135037>`_ 观察 `统计数据 <https://www.bilibili.com/video/BV1ib411f7YF/>`_ 推断得到。

.. 本节部分内容自一个资深的烧饼编写文档修改而来

参数意义
------------------------

    使用预定义好的抽卡模型须知参数（使用预定义好的抽卡模型已经可以满足大部分的计算需求了）

    - ``item_num`` 需求道具个数，由于 sphinx autodoc 的 `bug <https://github.com/sphinx-doc/sphinx/issues/9342>`_ 在下面没有显示

    - ``multi_dist`` 是否以列表返回获取 1-item_num 个道具的所有分布列，默认为False

    - ``item_pity`` 道具保底状态，通俗的叫法为水位、垫抽，默认为0

    - ``type_pity`` 定向选调的类型保底状态，默认为0，即该机制尚未开始触发；若为其它数，如20，那么就意味着定向选调机制已经垫了20抽，还有130抽就位于该保底机制的触发范围了（算上偏移量，实际为131抽）

    - ``calc_pull`` 采用伯努利模型时最高计算的抽数，高于此不计算（仅五星采用伯努利模型）

    使用已有的基础概率模型/明日方舟特有保底模型来自定义抽卡模型须知参数/常数（需要个性化定制抽卡需求才看，示例详见 `自定义抽卡模型 <#custom>`_ ）

    参数部分

    - ``type_pity_gap`` 保底类型触发抽数。目前已发现的保底类型有两种，分别是类型硬保底 :class:`~GGanalysis.AKHardPityModel` 和定向选调 :class:`~GGanalysis.AKDirectionalModel`

    - ``item_types`` 目标道具类别个数，可以简单理解为“UP了几个干员”

    - ``up_rate`` UP的目标道具所占比例，与``item_types``相配合。以标准寻访-轮换双UP池举例，50%UP两个六星干员，那么``up_rate``就为0.5，``item_types``就为2

    - ``type_pull_shift`` 保底类型偏移量，默认为0（无偏移），但明日方舟的保底实际触发抽数是保底线+1，因此在自定义模型时，该值为1。比如六星类型硬保底的保底线为200，但实际从201抽开始才能享受保底，往后偏移了1，因此这个保底类型偏移量就是1

    - ``total_item_types`` 全部目标道具的类别个数，也可以简单理解为“总共UP了几个干员”，用于满足计算“从x个干员中获得y个‘不同的’干员”的要求的参数，

    - ``collect_item`` 想要收集到的不同目标道具类别个数，与``total_item_types``相配合。以标准寻访-联合行动举例，总共UP了4个六星干员，而我想抽到其中3个，那么``total_item_types``就为4，``collect_item``就为3

    常数部分

    - ``PITY_6STAR`` 6星概率递增表，即从第1抽开始，至第99抽，每一抽获取到的干员是六星干员的概率。可以用于作为自定义模型的参数，指定计算的是六星干员还是五星干员。递增规律详见 `明日方舟抽卡系统解析 <https://www.bilibili.com/read/cv20251111?spm_id_from=333.999.0.0>`_

    - ``PITY_5STAR`` 5星概率递增表，概念及递增规律同上

    - ``P_5STAR_AVG`` 5星综合概率，由于五星干员没有类型硬保底机制，所以在计算时一般采用程序模拟出来的五星获取概率，而非概率递增表

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
.. _DirectionalModel:

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

其它使用示例
------------------------

自定义抽卡模型
>>>>>>>>>>>>>>>>>>
.. _custom:

.. attention::

    `AK_Limit_Model` 并非为默认公开的模型构造器，该构造器尚未加入新增的定向选调机制，这意味着使用该构造器构造轮换单UP池的模型是不符合实际的，其使用的 `CouponCollectorLayer` 当前为了计算效率只考虑集齐 1-k 种的情况，没有考虑集齐多套的需求，此后可能会新增支持多套的计算层。这个构造器接下来变化的可能性很大，因此当前不打算把这部分构造模型设为其他位置可以直接引用，如果想要在其他地方引用的话可以先临时复制代码出来本地使用，或是将 `AK_Limit_Model` 加入__all__公开列表进行调用。
    `AK_Limit_Model` 中的 `item_num` 并非表示每种道具集齐了 item_num 个，而是表示每次集齐后就清空，集齐了多少轮。
    `AK_Limit_Model` 构造器用于返回一个以用户输入的参数为基础数据的，要求在n个UP干员中获取到m个UP干员的抽卡模型（n >= m，基础数据包括 `total_item_types` 卡池中UP了几个干员、 `collect_item` 想要获取到几个UP干员等）

    `AKHardPityModel` 可以为某个 `FiniteDist` 类型的，无保底类型的分布载入类型硬保底机制。

    `AKDirectionalModel` 可以为某个 `FiniteDist` 类型的，无保底类型的分布载入定向选调机制。

    `p2dist` 是一个程序计算过程中的工具函数，用于将保底概率参数转化为分布列，一般用户无需单独使用该函数，仅作为指定干员星级的抽卡模型参数传入即可。

    构造器返回的是一个模型，而非分布。

    自定义抽卡模型时建议手动载入的模块/函数：`AKHardPityModel` `AKDirectionalModel` `AK_Limit_Model` `p2dist`

**联合行动池集齐三种UP六星干员**

.. imperfectmethod:: GGanalysis.games.arknights.gacha_model.AK_Limit_Model

.. code:: python

    import GGanalysis.games.arknights as AK
    triple_up_specific_6star = AK.AK_Limit_Model(AK.PITY_6STAR, 1, total_item_types=3, collect_item=3)
    dist = triple_up_specific_6star(item_pity=5) # （默认）期望集齐一轮，此前垫了5抽
    print('期望抽数为：{}'.format(dist.exp)) # 期望抽数为：188.63258247595024
    print('方差为：{}'.format(dist.var)) # 方差为：10416.175324956945
    print('100抽以内达成目标的概率为：{}%'.format(sum(dist[:100+1]) * 100)) # 100抽以内达成目标的概率为：16.390307170816875%

**定向寻访池获取特定六星干员**

.. automethod:: GGanalysis.basic_models.PityBernoulliModel
.. protectedmethod:: GGanalysis.games.arknights.gacha_model.AKHardPityModel
.. protectedmethod:: GGanalysis.games.arknights.gacha_model.p2dist

.. code:: python

    import GGanalysis as GG
    import GGanalysis.games.arknights as AK
    # 六星100%概率，UP三个，故抽到目标UP六星的概率为1/3
    triple_up_specific_6star = GG.PityBernoulliModel(AK.PITY_6STAR, 1 / 3) # 尚未证实定向寻访是否存在类型硬保底机制，保险起见，仅使用伯努利模型
    dist = triple_up_specific_6star(2) # 在定向寻访池期望抽到目标六星干员两次，此前没有垫抽
    print('期望抽数为：{}'.format(dist.exp)) # 期望抽数为：207.56732961125866

**双UP限定池获取特定权值提升的非UP六星干员**

    在70%概率UP双六星干员（以下简称为“主要UP干员”）的限定池中，获取权值以5倍提高的六星干员（以下简称为“次要UP干员”）的概率公式为：p = 0.3 / (5 * second_up_number + others) * 5
    其中，second_up_number为次要UP干员的数量，others为除了主要UP干员和次要UP干员外，其他准许获取的六星干员的数量。

.. automethod:: GGanalysis.basic_models.PityBernoulliModel

.. code:: python

    import GGanalysis as GG
    others = 71 # 假设除了主要UP干员和次要UP干员外，其他准许获取的六星干员的数量为71
    triple_second_up_specific_6star = GG.PityBernoulliModel(AK.PITY_6STAR, 0.3 / (5 * 3 + others) * 5) # 在当前卡池内，次要UP干员数量一般为3
    success_count = 3 # 期望抽到某个次要UP干员三次
    dist = triple_second_up_specific_6star(success_count, True) # `multi_dist` 为True表示以列表形式返回分布
    for i in range(1, success_count + 1):
        print(f"抽到第{i}个目标干员~期望抽数：{round(dist[i].exp, 2)}，方差：{round(dist[i].var, 2)}") # 结果保留两位小数

    # 抽到第1个目标干员~期望抽数：1983.42，方差：3890458.19
    # 抽到第2个目标干员~期望抽数：3966.84，方差：7780916.38
    # 抽到第3个目标干员~期望抽数：5950.26，方差：11671374.57

    # 不太建议计算非主要UP的干员的数据，分布会很长

**标准寻访-单UP池中集齐两种UP五星干员**

.. imperfectmethod:: GGanalysis.games.arknights.gacha_model.AK_Limit_Model

.. code:: python

    import GGanalysis.games.arknights as AK
    both_up_5star = AK.AK_Limit_Model(AK.PITY_5STAR, 0.5, total_item_types=2, collect_item=2)
    dist = both_up_5star()  # 期望在轮换单UP池中抽到两个UP的五星干员，此前没有垫抽
    print('期望抽数为：{}'.format(dist.exp))  # 期望抽数为：63.03402819816313

**定向选调附加**

.. attention::

    定向选调是仅针对标准寻访-单UP池设计的机制，错误地将之载入其它卡池类型的抽卡模型一定会得到错误的结果！

.. automethod:: GGanalysis.basic_models.PityBernoulliModel
.. protectedmethod:: GGanalysis.games.arknights.gacha_model.AKDirectionalModel
.. protectedmethod:: GGanalysis.games.arknights.gacha_model.p2dist

.. code:: python

    # 获取标准寻访-单UP6星（该模型已被定义好  `有定向选调获取单UP六星干员 <#DirectionalModel>`_ ，这里仅做说明使用）
    single_up_6star_without_directional = GG.PityBernoulliModel(AK.PITY_6STAR, 1 / 2)  # 无定向选调
    single_up_6star_has_directional = AK.AKDirectionalModel(single_up_6star_without_directional(1), AK.p2dist(AK.PITY_6STAR), type_pity_gap=150, item_types=1, up_rate=0.5)  # 载入定向选调
    dist = single_up_6star_has_directional(3)
    print('期望抽数为：{}'.format(dist.exp))  # 期望抽数为：199.2538696261363

**类型硬保底附加**

.. attention::

    目前已知的，存在类型硬保底的卡池类型为标准寻访-轮换双UP池，其它卡池暂无证据表明存在此机制，因此切勿盲目为其它卡池附加此机制。

.. automethod:: GGanalysis.basic_models.PityBernoulliModel
.. protectedmethod:: GGanalysis.games.arknights.gacha_model.AKHardPityModel
.. protectedmethod:: GGanalysis.games.arknights.gacha_model.p2dist

.. code:: python

    import GGanalysis as GG
    import GGanalysis.games.arknights as AK
    # 假设定向寻访池存在类型硬保底机制

    # 六星100%概率，UP三个，故抽到目标UP六星的概率为1/3
    triple_up_specific_6star_without_hard_pity = GG.PityBernoulliModel(AK.PITY_6STAR, 1 / 3) # 没有硬保底
    triple_up_specific_6star_has_hard_pity = AK.AKHardPityModel(triple_up_specific_6star_without_hard_pity(1), AK.p2dist(AK.PITY_6STAR), type_pity_gap=200, item_types=3, up_rate=1, type_pull_shift=1) # 载入硬保底
    dist = triple_up_specific_6star_has_hard_pity(2) # 在定向寻访池期望抽到目标六星干员两次，此前没有垫抽
    print('期望抽数为：{}'.format(dist.exp)) # 期望抽数为：207.0218117279958