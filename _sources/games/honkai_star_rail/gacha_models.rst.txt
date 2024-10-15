崩坏：星穹铁道抽卡模型
========================

GGanalysis 使用基本的抽卡模板模型结合 `基于1500万抽数据统计的崩坏：星穹铁道抽卡系统 <https://www.bilibili.com/video/BV1nt421w7FE/>`_ 定义了一系列可以直接取用的抽卡模型。

.. attention:: 

   崩坏：星穹铁道的四星保底不会被五星重置，但与五星耦合时仍会在综合概率上产生细微的影响。此处的模型没有考虑四星和五星的耦合。

   崩坏：星穹铁道的限定角色卡池与限定光锥卡池 **实际UP概率显著高于官方公示值** ，模型按照统计推理得到值建立。

   崩坏：星穹铁道常驻跃迁中具有“平稳机制”，即角色和光锥两种类型的保底，GGanalysis 包没有提供这类模型，有需要可以借用 `GGanalysislib 包 <https://github.com/OneBST/GGanalysislib>`_ 为原神定义的模型进行计算。角色活动跃迁及光锥活动跃迁中的四星道具也有此类机制，由于其在“UP机制”后生效，对于四星UP道具抽取可忽略。

参数意义
------------------------

    - ``item_num`` 需求物品个数，由于 sphinx autodoc 的 `bug <https://github.com/sphinx-doc/sphinx/issues/9342>`_ 在下面没有显示

    - ``multi_dist`` 是否以列表返回获取 1-item_num 个物品的所有分布列

    - ``item_pity`` 道具保底状态，通俗的叫法为水位、垫抽

    - ``up_pity`` UP道具保底状态，设为 1 即为玩家所说的大保底

基本模型
------------------------

**角色活动跃迁及常驻跃迁获得五星道具的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.common_5star

**角色活动跃迁及常驻跃迁获得四星道具的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.common_4star

角色活动跃迁模型
------------------------

**角色活动跃迁获得UP五星角色的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_5star_character

**角色活动跃迁获得任意UP四星角色的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_4star_character

**角色活动跃迁获得特定UP四星角色的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_4star_specific_character

.. code:: python

    import GGanalysis.games.honkai_star_rail as SR
    # 崩坏：星穹铁道角色池的计算
    print('角色池在垫了20抽，有大保底的情况下抽3个UP五星抽数的分布')
    dist_c = SR.up_5star_character(item_num=3, item_pity=20, up_pity=1)
    print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)

光锥活动跃迁模型
------------------------

**光锥活动跃迁获得五星光锥的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.common_5star_weapon

**光锥活动跃迁获得UP五星光锥的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_5star_weapon

**光锥活动跃迁获得四星光锥的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.common_4star_weapon

**光锥活动跃迁获得UP四星光锥的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_4star_weapon

**光锥活动跃迁获得特定UP四星光锥的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_4star_specific_weapon
