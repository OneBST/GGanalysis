基本概念
========================

FiniteDist 类型
------------------------

.. .. autoclass:: GGanalysis.FiniteDist
..     :special-members: __init__, __setitem__, __getitem__, __add__, __mul__, __rmul__, __truediv__, __pow__, __str__, __len__
..     :members:

:class:`~GGanalysis.FiniteDist` 是非常重要的类型，GGanalysis 工具包的计算功能建立在这个类型之上，大部分模型返回的值都以 ``FiniteDist`` 类型表示。

.. admonition:: 如何理解 FiniteDist 类型
   :class: note
   
   可以认为 GGanalysis 中的 FiniteDist 是对 numpy 数组的包装，核心是记录了离散随机变量的分布信息。

**创建新的 FiniteDist 类型**

FiniteDist 用于描述自然数位置上的有限长分布，用一个数组记录从0位置开始分布的概率。可以视为记录了一个样本空间为自然数的随机变量的信息。
可以使用列表、numpy 数组或者另一个 ``FiniteDist`` 来创建新的 ``FiniteDist``。

.. code:: Python

    import GGanalysis as gg
    import numpy as np

    dist_a = gg.FiniteDist([0, 0.5, 0.5])
    dist_b = gg.FiniteDist(np.array([0, 1]))
    dist_c = gg.FiniteDist(dist_a)

**从 FiniteDist 中提取信息**

``FiniteDist`` 类型的分布以 numpy 数组形式记录在 ``FiniteDist.dist`` 中。同时可以从 ``FiniteDist`` 类型中直接获取分布的基本属性，如期望和方差。

.. code:: Python

    print('dist_a 的分布数组', dist_a.dist)  # [0.  0.5 0.5]
    print('dist_a 的期望', dist_a.exp)  # 1.5
    print('dist_a 的方差', dist_a.var)  # 0.25

**用 FiniteDist 表达随机变量之和**

``FiniteDist`` 类型之间的 ``*`` 运算被定义为卷积，数学意义为这两个随机变量和的随机变量。

    在抽卡游戏中，可以认为是获取了 A 道具后再获取 B 道具，两个事件叠加后所需抽数的分布。

.. code:: Python

    dist_c = dist_a * dist_b
    print('混合后的分布', dist_c.dist)  # [0.  0.  0.5 0.5]

**用 FiniteDist 表达独立同分布随机变量之和**

``FiniteDist`` 类型也定义了 ``**`` 运算，返回分布为指定个数独立同分布随机变量分布相卷积的 ``FiniteDist`` 对象。

.. code:: Python

    dist_c = dist_b ** 3
    print('乘方后的分布', dist_c.dist)  # [0. 0. 0. 1.]

**对分布进行数量乘**

``FiniteDist`` 类型与数字类型之间的 ``*`` 运算是对 ``FiniteDist.dist`` 的简单数乘。请注意数量乘是为了方便进行一些操作时提供的运算，为满足归一性需要后续继续处理。

.. code:: Python

    dist_c = dist_a * 2
    print('数量乘后的分布数组', dist_c.dist)  # [0. 1. 1.]

**对分布进行数量加**

``FiniteDist`` 类型之间的 ``+`` 运算被定义为其分布数组按 0 位置对齐直接相加。请注意数量加是为了方便进行一些操作时提供的运算，为满足归一性需要后续继续处理。

.. code:: Python

    dist_c = dist_a + dist_b
    print('数量加的分布数组', dist_c.dist)  # [0, 1.5, 0.5]

CommonGachaModel 类型
------------------------

:class:`~GGanalysis.CommonGachaModel` 用于描述每次获得指定道具所需抽数分布都是独立的抽卡模型。
是大部分预定义抽卡模型的基类。
通过 ``CommonGachaModel`` 定义的类通常接收所需道具的个数，当前条件信息，最后以 FiniteDist 类型返回所需抽数分布。

``CommonGachaModel`` 定义时接受抽卡层的组合，以此构建按顺序复合各个抽卡层，自动推理出组合后的概率分布，以此快捷地构造出复杂的抽卡模型。当前支持的抽卡层有：

    1. ``Pity_layer`` 保底抽卡层，实现每抽获取物品的概率仅和当前至多多少抽没有获取过物品相关的抽卡模型。
    
    2. ``Bernoulli_layer`` 伯努利抽卡层，实现每次抽卡获取物品的概率都是相互独立并有同样概率的抽卡模型。
    
    3. ``Markov_layer`` 马尔可夫抽卡层，实现每次抽卡都按一定概率在状态图上进行转移的抽卡模型。保底抽卡层是马尔科夫抽卡层的特例。
    
    4. ``Coupon_Collector_layer`` 集齐道具层，实现每次抽卡随机获得某种代币，代币有若干不同种类，当集齐一定种类的代币后获得物品的抽卡模型。（注意：目前集齐道具层的功能已经可以使用，但还未经过充分的测试）

.. .. autoclass:: GGanalysis.FiniteDist
..     :members:

.. .. autoclass:: GGanalysis.CommonGachaModel
..     :members: