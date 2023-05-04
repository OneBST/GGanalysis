碧蓝航线抽卡模型
========================

.. attention:: 

   GGanalysis 提供的预定义模型没有考虑 200 抽天井的情况，如需考虑此情况请参照此 :ref:`示例代码 <azur_lane_hard_pity_example>` 。
   碧蓝航线的预定义抽卡模型按卡池 **彩 金 紫** 道具数量进行模型命名，例如1彩2金3紫模型命名为 ``model_1_2_3``，此例中预定义的彩、金、紫道具名称分别为 ``UR1`` ``SSR1`` ``SSR2`` ``SR1`` ``SR2`` ``SR3``。 
   使用时输入初始道具收集状态和目标道具收集状态，模型以 ``FiniteDist`` 类型返回所需抽数分布。

参数意义
------------------------

    - ``init_item`` 初始道具收集状态，一个包含了已经拥有哪些道具的字符串列表，由于 sphinx autodoc 的 `bug <https://github.com/sphinx-doc/sphinx/issues/9342>`_ 在下面没有显示

    - ``target_item`` 目标道具收集状态，一个包含了目标要收集哪些道具的字符串列表

预定义模型
------------------------

.. automethod:: GGanalysis.games.azur_lane.gacha_model.model_1_1_3

.. automethod:: GGanalysis.games.azur_lane.gacha_model.model_0_3_2

.. automethod:: GGanalysis.games.azur_lane.gacha_model.model_0_2_3

.. automethod:: GGanalysis.games.azur_lane.gacha_model.model_0_2_2

.. automethod:: GGanalysis.games.azur_lane.gacha_model.model_0_2_1

.. automethod:: GGanalysis.games.azur_lane.gacha_model.model_0_1_1

.. code:: python

    import GGanalysis.games.azur_lane as AL
    # 碧蓝航线2金3紫卡池在已有1金0紫的情况下集齐剩余1金及特定2紫的情况
    dist = AL.model_0_2_3(init_item=['SSR1'], target_item=['SSR1', 'SSR2', 'SR1', 'SR2'])
    print('期望为', dist.exp, '方差为', dist.var, '分布为', dist.dist)

.. _azur_lane_hard_pity_example:

处理天井示例代码
------------------------

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_4star_specific_character

.. code:: python

    import GGanalysis as gg
    import GGanalysis.games.azur_lane as AL
    # 对于 1_1_3 带200天井的情况，需要单独处理
    dist_0 = AL.model_1_1_3(target_item=['UR1', 'SSR1'])  # 未触发200天井时
    dist_1 = AL.model_1_1_3(target_item=['SSR1'])  # 触发200天井时
    cdf_0 = gg.dist2cdf(dist_0)
    cdf_1 = gg.dist2cdf(dist_1)
    cdf_0 = cdf_0[:min(len(cdf_0), len(cdf_1))]
    cdf_1 = cdf_1[:min(len(cdf_0), len(cdf_1))]
    cdf_0[200:] = cdf_1[200:]
    # 此时 cdf_0 就是含天井的累积概率密度函数 

