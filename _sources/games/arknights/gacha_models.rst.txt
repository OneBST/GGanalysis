.. _arknights_gacha_model:

明日方舟抽卡模型
========================

GGanalysis 使用基本的抽卡模板模型结合 `明日方舟抽卡系统参数 <https://www.bilibili.com/read/cv20251111>`_ 定义了一系列可以直接取用的抽卡模型。需要注意的是明日方舟的抽卡系统模型的确定程度并没有很高，使用时需要注意。

此外，还针对性编写了如下模板模型：

    适用于计算 `定向寻访 <https://www.bilibili.com/read/cv22596510>`_ 时获得特定UP六星角色的模型
    :class:`~GGanalysis.games.arknights.AKDirectionalModel`

    适用于计算通过 `统计数据 <https://www.bilibili.com/video/BV1ib411f7YF/>`_ 发现的类型硬保底的模型
    :class:`~GGanalysis.games.arknights.AKHardPityModel`

    适用于计算集齐多种六星的模型（不考虑300井、定向寻访及类型硬保底机制）
    :class:`~GGanalysis.games.arknights.AK_Limit_Model` 

.. 本节部分内容自一个资深的烧饼编写文档修改而来

参数意义
------------------------

    - ``item_num`` 需求物品个数，由于 sphinx autodoc 的 `bug <https://github.com/sphinx-doc/sphinx/issues/9342>`_ 在下面没有显示

    - ``multi_dist`` 是否以列表返回获取 1-item_num 个物品的所有分布列

    - ``item_pity`` 道具保底状态，通俗的叫法为水位、垫抽

    - ``type_pity`` 定向选调的类型保底状态

    - ``calc_pull`` 采用伯努利模型时最高计算的抽数，高于此不计算（仅五星采用伯努利模型）

六星模型
------------------------

**获取任意六星角色**

.. automethod:: GGanalysis.games.arknights.gacha_model.common_6star

.. code:: python

    import GGanalysis.games.arknights as AK
    dist = AK.common_6star(item_num=1)
    print('抽到六星的期望抽数为：{}'.format(dist.exp))  # 34.59455493520977

**无定向选调获取单UP六星**

.. automethod:: GGanalysis.games.arknights.gacha_model.single_up_6star_old

**有定向选调获取单UP六星**

.. automethod:: GGanalysis.games.arknights.gacha_model.single_up_6star

.. code:: python

    import GGanalysis.games.arknights as AK
    dist = AK.single_up_6star(item_num=1, item_pity=0, type_pity=0)
    print('4.6寻访机制更新后，无水位时抽到单up六星的期望抽数为：{}'.format(dist.exp))
    
.. container:: output stream stdout

    ::

        4.6寻访机制更新后，无水位时抽到单up六星的期望抽数为：66.16056206529494

**无类型保底轮换池获取特定六星**

.. automethod:: GGanalysis.games.arknights.gacha_model.dual_up_specific_6star_old

**有类型保底时轮换池获取特定六星**

    轮换池具有类型保底，即首个 201 抽无 UP 六星后，下个六星必为 UP 六星其一；首个 401 抽无特定 UP 六星后，下个六星必为特定 UP 六星。每个卡池仅生效一轮。

    该规律由 `一个资深的烧饼- <https://space.bilibili.com/456135037>`_ 观察 `统计数据 <https://www.bilibili.com/video/BV1ib411f7YF/>`_ 推断得到。

.. automethod:: GGanalysis.games.arknights.gacha_model.dual_up_specific_6star

.. code:: python

    import GGanalysis.games.arknights as AK
    dist = AK.dual_up_specific_6star(item_num=1)
    print('准备100抽，从轮换池捞出玛恩纳的概率只有：{}%'.format(sum(dist[:100+1]) * 100))

.. container:: output stream stdout

    ::

        准备100抽，从轮换池捞出玛恩纳的概率只有：49.60442859476116%

**限定池获取特定六星角色**

.. automethod:: GGanalysis.games.arknights.gacha_model.limited_up_6star

    
    需要注意的是，此模型返回的结果是不考虑井的分布。如需考虑井需要自行进行一定后处理。


.. code:: python

    import GGanalysis.games.arknights as AK
    dist = AK.limited_up_6star(item_num=5)
    print('一井满潜限定的概率：{}%'.format(sum(dist_4[:300+1]) * 100))

.. container:: output stream stdout

    ::

        一井满潜限定的概率：14.881994954229667%

**限定池集齐两种UP六星**

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

   明日方舟五星角色实际上有概率递增的星级保底机制，但其保底进度会被六星重置。这里对五星模型采用了近似，认为其是一个概率为考虑了概率递增的伯努利模型。另外，此处提供的五星模型也没有考虑类型保底。
   
   此外明日方舟五星模型没有采用 :class:`~GGanalysis.BernoulliLayer` 构建模型，而是直接采用了 :class:`~GGanalysis.BernoulliGachaModel` ，当设置 ``calc_pull`` 太低时，返回的分布概率和可能距离 1 有相当大差距，需要适当设高。

**获取任意五星角色**

.. automethod:: GGanalysis.games.arknights.gacha_model.common_5star

**获取单UP五星角色**

.. automethod:: GGanalysis.games.arknights.gacha_model.single_up_specific_5star

**获取双UP中特定五星角色**

.. automethod:: GGanalysis.games.arknights.gacha_model.dual_up_specific_5star

**获取三UP中特定五星角色**

.. automethod:: GGanalysis.games.arknights.gacha_model.triple_up_specific_5star


其它使用示例
------------------------