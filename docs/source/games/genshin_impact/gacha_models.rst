.. _genshin_gacha_model:

原神抽卡模型
========================

**GGanalysis** 使用基本的抽卡模板模型结合 `原神抽卡系统参数 <https://www.bilibili.com/read/cv10468091>`_ 定义了一系列可以直接取用的抽卡模型。

此外，还针对性编写了如下模板模型：

    适用于计算武器活动祈愿定轨时获取道具问题的模型
    :class:`~GGanalysis.games.genshin_impact.Genshin5starEPWeaponModel`

    适用于计算在活动祈愿中获得常驻祈愿五星/四星道具的模型
    :class:`~GGanalysis.games.genshin_impact.GenshinCommon5starInUPpoolModel` 

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

其他模型
------------------------

**从角色活动祈愿中获取位于常驻祈愿的特定五星角色的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.stander_5star_character_in_up

**从武器活动祈愿中获取位于常驻祈愿的特定五星武器的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.stander_5star_weapon_in_up