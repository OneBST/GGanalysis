自定义抽卡模型
========================

可以继承 :class:`~GGanalysis.CommonGachaModel` 并组合不同抽卡层来设计自己的抽卡模型。

.. admonition:: 抽卡系统例子
   :class: note
   
   假设有一个游戏有这样的抽卡系统：每抽有 10% 概率获得道具， 90% 概率获得垃圾。
   当连续 3 抽都没有获得道具时，获得道具的概率上升到 60%，
   当连续 4 抽都没有获得道具时，获得道具的概率上升到 100%。
   道具中一共有 5 种不同类别，每种类别均分概率。

这个例子中当一定抽数没有获得道具时，下次获得道具的概率会逐渐上升，直到概率上升为 100%。将这种模型称为软保底模型。
同时注意到，这个例子中获取了道具后，道具可能是 5 种道具中的任意一种，将这种等概率选择的模型称为伯努利模型。
而如果在自定义的抽卡模型中将保底抽卡层和伯努利抽卡层按顺序结合起来，就可以建立获取指定类别的道具的模型了。

.. code:: Python
    
    import GGanalysis as gg
    # 定义一个星级带软保底，每个星级内有5种物品，想要获得其中特定一种的模型
    # 定义软保底概率上升表，第1-3抽概率0.1，第4抽概率0.6，第5抽保底
    pity_p = [0, 0.1, 0.1, 0.1, 0.6, 1]

    # 采用预定义的保底伯努利抽卡类
    gacha_model = gg.PityBernoulliModel(pity_p, 1/5)
    # 根据定义的类计算从零开始获取一个道具的分布，由于可能永远获得不了道具，分布是截断的
    dist = gacha_model(item_num=1, pull_state=0)

    # 从头定义抽卡类
    # 保底伯努利抽卡类
    class MyModel(gg.CommonGachaModel):
        # 限制期望的误差比例为 1e-8，达不到精度时分布截断位置为 1e5
        def __init__(self, pity_p, p, e_error = 1e-8, max_dist_len=1e5) -> None:
            super().__init__()
            # 增加保底抽卡层
            self.layers.append(gg.PityLayer(pity_p))
            # 增加伯努利抽卡层
            self.layers.append(gg.BernoulliLayer(p, e_error, max_dist_len))
    # 根据定义的类计算从零开始获取一个道具的分布，由于可能永远获得不了道具，分布是截断的
    gacha_model = MyModel(pity_p, 1/5)
    # 关于 pull_state 等条件输入，如有需要可以参考预置类中 __call__ 和 _build_parameter_list 的写法
    dist = gacha_model(item_num=1)