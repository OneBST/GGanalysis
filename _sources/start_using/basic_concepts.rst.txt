基本概念
========================

FiniteDist 类型
------------------------

:class:`~GGanalysis.FiniteDist` 是非常重要的类型，GGanalysis 工具包的计算功能建立在这个类型之上。
FiniteDist 用于描述自然数位置上的有限长分布，用一个数组记录从0位置开始分布的概率。
可以视为记录了一个样本空间为自然数的随机变量的信息。
可以使用列表、numpy数组或者另一个 FiniteDist 来创建新的 FiniteDist。
FiniteDist 类型的分布以 numpy 数组形式记录在 ``FiniteDist.dist`` 中。

.. code:: Python

    import GGanalysis as gg
    import numpy as np

    dist_a = gg.FiniteDist([0, 0.5, 0.5])
    dist_b = gg.FiniteDist(np.array([0, 1]))
    dist_c = gg.FiniteDist(dist_a)

FiniteDist 类型可以直接计算分布的基本属性，如期望和方差。

.. code:: Python

    print('dist_a 的期望', dist_a.exp)  # 1.5
    print('dist_a 的方差', dist_a.var)  # 0.25

FiniteDist 类型之间的 ``*`` 运算被定义为卷积，数学意义为这两个随机变量和的随机变量。
在抽卡游戏中，可以认为是获取了 A 道具后再获取 B 道具，两个事件叠加后所需抽数的分布。

.. code:: Python

    dist_c = dist_a * dist_b
    print('混合后的分布', dist_c.dist)  # [0.  0.  0.5 0.5]

FiniteDist 类型与数字类型之间的 ``*`` 运算是对 ``FiniteDist.dist`` 的简单数乘。
请注意数乘是为了方便进行一些操作时提供的运算，乘以非 1 的数字后不再满足归一性，需要后续继续处理。

.. code:: Python

    dist_c = dist_a * 2
    print('数乘后的分布', dist_c.dist)  # [0. 1. 1.]

FiniteDist 类型也定义了 ``**`` 运算，
返回分布为指定个数自身分布相卷积的 FiniteDist 对象

.. code:: Python

    dist_c = dist_b ** 3
    print('乘方后的分布', dist_c.dist)  # [0. 0. 0. 1.]

CommonGachaModel 类型
------------------------

:class:`~GGanalysis.CommonGachaModel` 用于描述每次获得指定道具所需抽数分布都是独立的抽卡模型。
是大部分预定义抽卡模型的基类。
通过 CommonGachaModel 定义的类通常接收所需道具的个数，当前条件信息，最后以 FiniteDist 类型返回所需抽数分布。

CommonGachaModel 定义时接受抽卡层的组合，以此构建按顺序复合各个抽卡层，以此快捷地构造出复杂的抽卡模型。当前支持的抽卡层有：

    1. ``Pity_layer`` 保底抽卡层，实现每抽获取物品的概率仅和当前至多多少抽没有获取过物品相关的抽卡模型。
    
    2. ``Bernoulli_layer`` 伯努利抽卡层，实现每次抽卡获取物品的概率都是相互独立并有同样概率的抽卡模型。
    
    3. ``Markov_layer`` 马尔可夫抽卡层，实现每次抽卡都按一定概率在状态图上进行转移的抽卡模型。保底抽卡层是马尔科夫抽卡层的特例。
    
    4. ``Coupon_Collector_layer`` 集齐道具层，实现每次抽卡随机获得某种代币，代币有若干不同种类，当集齐一定种类的代币后获得物品的抽卡模型。（注意：目前集齐道具层的功能已经可以使用，但还未经过充分的测试）

.. .. autoclass:: GGanalysis.FiniteDist
..     :members:

.. .. autoclass:: GGanalysis.CommonGachaModel
..     :members: