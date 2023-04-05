.. _genshin_gacha_model:

原神抽卡模型
========================

GGanalysis 使用基本的抽卡模板模型结合 `原神抽卡系统参数 <https://www.bilibili.com/read/cv10468091>`_ 定义了一系列可以直接取用的抽卡模型。

此外，还针对性编写了如下模板模型：

    适用于计算武器活动祈愿定轨时获取道具问题的模型
    :class:`~GGanalysis.games.genshin_impact.Genshin5starEPWeaponModel`

    适用于计算在活动祈愿中获得常驻祈愿五星/四星道具的模型
    :class:`~GGanalysis.games.genshin_impact.GenshinCommon5starInUPpoolModel` 

参数意义
------------------------

    - ``item_num`` 需求物品个数，由于 sphinx autodoc 的 `bug <https://github.com/sphinx-doc/sphinx/issues/9342>`_ 在下面没有显示

    - ``multi_dist`` 是否以列表返回获取 1-item_num 个物品的所有分布列

    - ``item_pity`` 道具保底状态，通俗的叫法为水位、垫抽

    - ``up_pity`` UP道具保底状态，设为 1 即为玩家所说的大保底

    - ``calc_pull`` 采用伯努利模型时最高计算的抽数，高于此不计算（仅五星采用伯努利模型）

.. attention:: 

   原神的四星保底不会被五星重置，但与五星耦合时仍会在综合概率上产生细微的影响。此处的模型没有考虑四星和五星的耦合。

   原神常驻祈愿中具有“平稳机制”，即角色和武器两种类型的保底，GGanalysis 包没有提供这类模型，有需要可以使用 `GGanalysislib 包 <https://github.com/OneBST/GGanalysislib>`_ 。

基本模型
------------------------

**角色活动祈愿及常驻祈愿获得五星道具的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.common_5star

**角色活动祈愿及常驻祈愿获得四星道具的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.common_4star

角色活动祈愿模型
------------------------

**角色活动祈愿获得UP五星角色的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_5star_character

**角色活动祈愿获得任意UP四星角色的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_4star_character

**角色活动祈愿获得特定UP四星角色的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_4star_specific_character

.. code:: python

    import GGanalysis.games.genshin_impact as GI
    # 原神角色池的计算
    print('角色池在垫了20抽，有大保底的情况下抽3个UP五星抽数的分布')
    dist_c = GI.up_5star_character(item_num=3, item_pity=20, up_pity=1)
    print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)

武器活动祈愿模型
------------------------

**武器活动祈愿获得五星武器的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.common_5star_weapon

**武器活动祈愿获得UP五星武器的模型**
    
    注意此模型建模的是获得任意一个UP五星武器即满足要求的情况

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_5star_weapon

**武器活动祈愿无定轨情况下获得特定UP五星武器的模型**

    注意此模型建模的是无定轨情况下获得特定UP五星武器

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_5star_specific_weapon

**武器活动祈愿定轨情况下获得特定UP五星武器的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_5star_ep_weapon

**武器活动祈愿获得四星武器的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.common_4star_weapon

**武器活动祈愿获得UP四星武器的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_4star_weapon

**武器活动祈愿获得特定UP四星武器的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_4star_specific_weapon

.. code:: python

    import GGanalysis.games.genshin_impact as GI
    print('武器池池在垫了30抽，有大保底，命定值为1的情况下抽1个UP五星抽数的分布')
    dist_w = GI.up_5star_ep_weapon(item_num=1, item_pity=30, up_pity=1, fate_point=1)
    print('期望为', dist_w.exp, '方差为', dist_w.var, '分布为', dist_w.dist)

其它模型
------------------------

**从角色活动祈愿中获取位于常驻祈愿的特定五星角色的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.stander_5star_character_in_up

**从武器活动祈愿中获取位于常驻祈愿的特定五星武器的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.stander_5star_weapon_in_up

其它使用示例
------------------------

.. code:: python

    # 联合角色池和武器池
    print('在前述条件下抽3个UP五星角色，1个特定UP武器所需抽数分布')
    dist_c_w = dist_c * dist_w
    print('期望为', dist_c_w.exp, '方差为', dist_c_w.var, '分布为', dist_c_w.dist)

    # 对比玩家运气
    dist_c = GI.up_5star_character(item_num=10)
    dist_w = GI.up_5star_ep_weapon(item_num=3)
    print('在同样抽了10个UP五星角色，3个特定UP五星武器的玩家中，仅花费1000抽的玩家排名前', str(round(100*sum((dist_c * dist_w)[:1001]), 2))+'%')