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
    
    # 普池双UP的计算
    # item_num 是要抽多少个
    # pull_state 是当前垫了多少抽，从零开始填0就行
    dist_c = AK.dual_up_specific_6star(item_num=3, pull_state=20)
    print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)
    
    # 普池单UP的计算
    dist_c = AK.single_up_6star(item_num=1, pull_state=10) # 目标：获取UP六星1次，当前已垫了20抽
    print('40抽以内达成目标的概率为', sum(dist_c[:40+1]))
    print('在20-60抽之间达成目标的概率为', sum(dist_c[20:60+1])) # 
    
    # 普池三UP（定向寻访）的计算 
    # PITY_6STAR 是六星概率递增表，意为当前要计算抽六星干员的相关数据
    # 1 为目标物品（UP干员）的UP概率占比，在定向寻访中，100%UP三个六星干员，所以这里就是1
    # total_item_types 是总的物品类别数量，在这里代表定向寻访池有几个UP六星干员，3就表示定向寻访UP了三个六星干员
    # collect_item 是需要获取到的物品类别数量，也就是你想要抽到几个“不同的”UP六星干员，collect_item的值不能大于total_item_types的值
    # 当前需求：抽定向寻访池的六星干员，要求获得在3个UP六星干员之中抽到2个的数据
    dist_c = AK.AK_Limit_Model(AK.PITY_6STAR, 1, total_item_types=3, collect_item=2)
    
    # 普池四UP（联合行动）的计算
    # 当前需求：抽联合行动池的六星干员，要求获得抽到所有的UP六星干员的数据
    dist_c = AK.AK_Limit_Model(AK.PITY_6STAR, 1, total_item_types=4, collect_item=4) # 其他类似需求也可以使用 AK_Limit_Model 模型类推
    
    # 限定池双UP的计算
    # 1 为期望抽到双UP限定池所有UP六星干员的次数，这里为一次，即抽到两个UP六星就结束的需求
    # 0 为已垫的抽数，这里为0，也就是之前没有垫抽数
    dist_c = AK.both_up_6star(1, 0) # 获取抽到双UP限定池所有UP六星干员“一次”的数据
    # 6 为期望抽到双UP限定池中某一个UP六星干员的次数，这里为6次，即将该UP干员抽至满潜能的需求
    # 20 为已垫的抽数，这里为20，也就是之前已经垫了20抽
    dist_c = AK.limited_up_6star(6, 20) # 获取抽到双UP限定池中某一个UP六星干员六次的数据。需要注意，limited_up_6star这个模型特指的是“70%UP双六星干员”的限定池模型，该卡池中，每个UP的六星干员被分配到的概率为35%
    

更多预定义好的抽卡模型可以在 ``GGanalysis\games\gamename\gacha_model`` 内找到。
