快速可视化
========================

绘制简略的概率质量函数图及累积质量函数图
---------------------------------------------

.. code:: Python
        
    import GGanalysis.games.genshin_impact as GI
    # 获得原神抽一个UP五星角色+定轨抽特定UP五星武器的分布
    dist = GI.up_5star_character(item_num=1) * GI.up_5star_ep_weapon(item_num=1)
    # 导入绘图模块
    from GGanalysis.gacha_plot import DrawDistribution
    fig = DrawDistribution(dist, dpi=72, show_description=True)
    fig.draw_two_graph()