使用预定义的抽卡模型
========================

预定义的抽卡模型多以 :class:`~GGanalysis.CommonGachaModel` 为基类。
根据输入信息返回 :class:`~GGanalysis.FiniteDist` 类型的结果。

以原神为例说明如何使用预先定义好的抽卡模型。

.. code:: Python

    # 导入预定义好的原神模块
    import GGanalysis.games.genshin_impact as GI
    # 原神角色池的计算
    dist_c = GI.up_5star_character(item_num=3, pull_state=20, up_guarantee=1)
    print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)

以明日方舟为例说明如何使用预先定义好的抽卡模型。

.. code:: Python

    # 计算抽卡所需抽数分布律 以明日方舟为例
    import GGanalysis.games.arknights as AK
    # 普池双UP的计算 item_num是要抽多少个 pull_state是当前垫了多少抽，从零开始填0就行
    dist_c = AK.dual_up_specific_6star(item_num=3, pull_state=20)
    print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)

更多预定义好的抽卡模型可以在 ``GGanalysis\games\gamename\gacha_model`` 内找到。