绝区零抽卡模型
========================

GGanalysis 使用基本的抽卡模板模型结合 `基于700万抽数据统计的绝区零抽卡系统 <https://www.bilibili.com/video/BV1oW42197RR/>`_ 定义了一系列可以直接取用的抽卡模型。

.. attention:: 

   绝区零的 **A级保底会被S级重置**，与S级耦合时会对A级在综合概率上产生明显的影响。
   单独的A级分布模型没有考虑A级和S级的耦合，计算得到A级概率是偏高的。计算S级和A级耦合后情况的代码位于 `此处 <https://github.com/OneBST/GGanalysis/blob/main/GGanalysis/games/zenless_zone_zero/stationary_p.py>`_ 。

   绝区零的常驻卡池中具有和原神与崩坏：星穹铁道都不一样的“平稳机制”，虽然也保证能在有限抽数内必定角色和武器两种类型的道具，
   但绝区零根据抽到的道具数而不是已经投入的抽数进行判断。
   若当前已有连续2个同类别S级物品，下个S级物品是另一类别的概率会大幅提高，观测到至多连续出3个同类别S级物品。
   对于A级物品则为当已有连续4个同类别A级物品，下个A级物品是另一类别的概率会大幅提高，观测到至多连续出5个同类别A级物品。

参数意义
------------------------

    - ``item_num`` 需求物品个数，由于 sphinx autodoc 的 `bug <https://github.com/sphinx-doc/sphinx/issues/9342>`_ 在下面没有显示

    - ``multi_dist`` 是否以列表返回获取 1-item_num 个物品的所有分布列

    - ``item_pity`` 道具保底状态，通俗的叫法为水位、垫抽

    - ``up_pity`` UP道具保底状态，设为 1 即为玩家所说的大保底

基本模型
------------------------

**角色池及常驻池获得S级道具的模型**

.. automethod:: GGanalysis.games.zenless_zone_zero.gacha_model.common_5star

**角色池及常驻池获得A级道具的模型**

.. automethod:: GGanalysis.games.zenless_zone_zero.gacha_model.common_4star

角色池模型
------------------------

**角色池获得UPS级角色的模型**

.. automethod:: GGanalysis.games.zenless_zone_zero.gacha_model.up_5star_character

**角色池获得任意UPA级角色的模型**

.. automethod:: GGanalysis.games.zenless_zone_zero.gacha_model.up_4star_character

**角色池获得特定UPA级角色的模型**

.. automethod:: GGanalysis.games.zenless_zone_zero.gacha_model.up_4star_specific_character

.. code:: python

    import GGanalysis.games.zenless_zone_zero as SR
    # 绝区零角色池的计算
    print('角色池在垫了20抽，有大保底的情况下抽3个UPS级抽数的分布')
    dist_c = SR.up_5star_character(item_num=3, item_pity=20, up_pity=1)
    print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)

武器池模型
------------------------

**武器池获得S级武器的模型**

.. automethod:: GGanalysis.games.zenless_zone_zero.gacha_model.common_5star_weapon

**武器池获得UPS级武器的模型**

.. automethod:: GGanalysis.games.zenless_zone_zero.gacha_model.up_5star_weapon

**武器池获得A级武器的模型**

.. automethod:: GGanalysis.games.zenless_zone_zero.gacha_model.common_4star_weapon

**武器池获得UPA级武器的模型**

.. automethod:: GGanalysis.games.zenless_zone_zero.gacha_model.up_4star_weapon

**武器池获得特定UPA级武器的模型**

.. automethod:: GGanalysis.games.zenless_zone_zero.gacha_model.up_4star_specific_weapon
