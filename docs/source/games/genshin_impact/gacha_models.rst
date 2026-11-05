.. _genshin_gacha_model:

原神抽卡模型
========================

GGanalysis 使用基本的抽卡模板模型结合 `原神抽卡系统参数 <https://www.bilibili.com/read/cv10468091>`_ 定义了一系列可以直接取用的抽卡模型。

此外，还针对性编写了如下模板模型：

    适用于计算5.0版本前武器活动祈愿定轨时获取道具问题的模型
    :class:`~GGanalysis.games.genshin_impact.gacha_model.ClassicGenshin5starEPWeaponModel`

    适用于计算在活动祈愿中获得常驻祈愿五星/四星道具的模型
    :class:`~GGanalysis.games.genshin_impact.ClassicGenshinCommon5starInUPpoolModel`

.. attention:: 

   原神的四星保底不会被五星重置，但与五星耦合时仍会在综合概率上产生细微的影响。此处的模型没有考虑四星和五星的耦合。

   常驻五星角色、武器类别模型已包含“平稳机制”；普通 ``common_5star``
   仍表示任意五星。四星模型尚未包含类别平稳机制及五星耦合。
   明光双计数器和类别权重均为机制猜想/拟合模型，不代表官方确认参数。

参数意义
------------------------

    - ``item_num`` 需求物品个数，由于 sphinx autodoc 的 `bug <https://github.com/sphinx-doc/sphinx/issues/9342>`_ 在下面没有显示

    - ``multi_dist`` 是否以列表返回获取 0-item_num 个物品的所有分布列

    - ``item_pity`` 道具保底状态，通俗的叫法为水位、垫抽

    - ``up_pity`` UP道具保底状态，设为 1 即为玩家所说的大保底

    - ``ep_pity`` 当前武器定轨模型的命定值状态；经典武器模型使用 ``fate_point``，二者不可混用

    - ``cr_counter`` 「捕获明光」计数器，按当前模型状态定义取值；不等同于简单连歪次数

    - ``cr_counter_b`` 明光 B 计数器，默认 3；仅知道 A 不能推断 B

基本模型
------------------------

**角色活动祈愿及常驻祈愿获得五星道具的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.common_5star

**角色活动祈愿及常驻祈愿获得四星道具的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.common_4star

角色活动祈愿模型
------------------------

**角色活动祈愿5.0版本后获得UP五星角色的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_5star_character

**角色活动祈愿5.0版本前获得UP五星角色的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.classic_up_5star_character

**角色活动祈愿获得任意UP四星角色的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_4star_character

**角色活动祈愿获得特定UP四星角色的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_4star_specific_character

.. code:: python

    import GGanalysis.games.genshin_impact as GI
    # 原神角色池的计算
    print('角色池在垫了20抽，有大保底，捕获明光计数器为2的情况下抽3个UP五星抽数的分布')
    dist_c = GI.up_5star_character(item_num=3, item_pity=20, up_pity=1, cr_counter=2)
    print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)

捕获明光双计数器
------------------------

初始计数器为 A=1、B=3。普通小保底不歪令 A 减 1、B 减 2，最低为零；
歪令 A 加 1、B 加 3；触发明光后重置为 (1,3)。大保底只清除保证标记，
不改变两个计数器。A=3 时必定直接触发明光，其余 A 值下概率只由 B 决定：
B=0～5 为零，6、7、8、9 分别为 1%、5%、25%、99%，B>=10 为 100%。
B 概率为用户提供的拟合估计，保留在 ``CR_B_P`` 单处定义。

这里“触发概率”是直接明光分支的概率 c，而不是原本会歪时的救回概率。
普通不歪、歪各占剩余概率 (1-c)/2；A、B 同时由结果更新，因此状态分布
存在相关性，不能分别求分布后相乘。自定义 ``cr_p``、``cr_b_p`` 时取两者
较大值；默认 A 表只有强制触发项。

模型在五星层使用 ``HitProcessAnalysis``，每件 UP 最多消耗两个五星，
保留每次命中后的 (A,B) 联合状态。得到五星件数分布后，组合普通五星
等待分布；只有首个五星使用 ``item_pity`` 条件分布。不会将 UP 周期当作 IID。

.. code-block:: python

   import GGanalysis.games.genshin_impact as GI

   dist = GI.up_5star_character(
       item_num=3, item_pity=20, up_pity=0, cr_counter=2, cr_counter_b=8,
   )
   print(dist.exp, dist.var)
   p = GI.up_5star_character.small_pity_up_rate()
   print(p)  # 默认拟合参数约为 0.5526916771149277

``small_pity_up_rate`` 返回长期小保底 UP 胜率，不计大保底，不是指定当前
计数器下的下一次胜率。若需要含大保底的五星 UP 比例，可自行转换为 1/(2-p)。
指定初态的首件期望不能用长期概率倒数代替。

.. autoclass:: GGanalysis.games.genshin_impact.CapturingRadianceModel
   :members: small_pity_up_rate

常驻五星类别模型
------------------------

``standard_5star_character`` 计算任意五星角色的等待分布；
``standard_5star_weapon`` 计算任意五星武器。二者共用
`原神抽卡全机制总结 <https://www.bilibili.com/opus/506065155991936609>`_
中的类别平稳规则，不是指定某个角色/武器的模型。

先按普通五星保底概率确定是否出五星，再选择角色/武器类别。若此前有 d 抽
未获得该类别，则本次使用等待位置 i=d+1，权重为 30+300*max(i-146,0)。
两类别权重轮盘上限为 10000，较大权重优先，不能在超过上限时仍简单除以权重和。
已有 179 抽未获得某类时，下个五星必为该类。因此达到 179 的类别进度可
精确合并，不是尾部截断；不能将此规则理解为连续两个同类五星后的保底。

``item_pity`` 为此前未获得任意五星的抽数；``character_pity`` 和
``weapon_pity`` 分别为未获得五星角色/武器的抽数，较小者必须等于 item_pity。
未指定的类别进度默认等于 item_pity，采用两类进度同时从零开始累计的初始假设，
不会自动使用稳态。每抽两类进度都增加，只在获得同类别五星时归零。

.. code-block:: python

   dist = GI.standard_5star_character(
       item_num=1, item_pity=20, character_pity=100, weapon_pity=20,
   )
   print(dist.exp, dist.var)
   # 角色/武器互换，同一个机制实现。
   weapon_dist = GI.standard_5star_weapon(
       item_num=1, item_pity=20, character_pity=20, weapon_pity=100,
   )
   multi = GI.standard_5star_character(item_num=2, multi_dist=True)

内部编码两类进度，省去可由较小值确定的五星水位，使用稀疏逐抽转移。
目标类别命中后保留另一类别进度，多件通过状态周期核计算，不能直接对首件
边缘分布求卷积幂。单件只做稀疏递推，不构造稠密周期核；多件才按需构造。
默认五星表下，任意初态首件最多 269 抽，每件周期也覆盖至 269 抽，无尾部补概率。

验证包含独立五星结果枚举、角色/武器对称性、179 抽强制类别边界、
文章等待位置索引，以及多件周期卷积与完整逐抽递推的确定性交叉对照。

.. autoclass:: GGanalysis.games.genshin_impact.StandardGenshin5starModel
   :members: __call__

武器活动祈愿模型
------------------------

**武器活动祈愿获得五星武器的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.common_5star_weapon

**武器活动祈愿获得UP五星武器的模型**
    
    注意此模型建模的是获得任意一个UP五星武器即满足要求的情况

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.up_5star_weapon

**武器活动祈愿无定轨情况下获得特定UP五星武器的模型**

    注意此模型建模的是2.0前无定轨情况下获得特定UP五星武器的情况

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.classic_up_5star_specific_weapon

**武器活动祈愿5.0版本后定轨情况下获得特定UP五星武器的模型**

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
    dist_w = GI.up_5star_ep_weapon(item_num=1, item_pity=30, up_pity=1, ep_pity=1)
    print('期望为', dist_w.exp, '方差为', dist_w.var, '分布为', dist_w.dist)

其它模型
------------------------

**5.0前从角色活动祈愿中获取位于常驻祈愿的特定五星角色的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.classic_stander_5star_character_in_up

**5.0前从武器活动祈愿中获取位于常驻祈愿的特定五星武器的模型**

.. automethod:: GGanalysis.games.genshin_impact.gacha_model.classic_stander_5star_weapon_in_up

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
