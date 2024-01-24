平稳分布时概率的计算工具
========================

使用转移矩阵方法计算复合类型保底的概率
----------------------------------------

.. code:: Python
    
    # 以明日方舟为例，计算明日方舟六星、五星、四星、三星物品耦合后，各类物品的综合概率
    import GGanalysis as gg
    import GGanalysis.games.arknights as AK
    # 按优先级将道具概率表组成列表
    item_p_list = [AK.PITY_6STAR, AK.PITY_5STAR, [0, 0.5], [0, 0.4]]
    AK_probe = gg.PriorityPitySystem(item_p_list, extra_state=1, remove_pity=True)
    # 打印结果，转移矩阵的计算可能比较慢
    print(AK_probe.get_stationary_p())  # [0.02890628 0.08948246 0.49993432 0.38167693]

使用迭代方法计算平稳分布后n连抽获得k个道具概率
--------------------------------------------------

.. code:: Python
    
    # 以原神10连获得多个五星道具为例
    import GGanalysis as gg
    import GGanalysis.games.genshin_impact as GI
    # 获得平稳后进行10连抽获得k个五星道具的分布
    ans = gg.multi_item_rarity(GI.PITY_5STAR, 10)
