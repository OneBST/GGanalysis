抽卡层
======================================

定义位置：``GGanalysis/gacha_layers.py``。抽卡层将“获得若干底层道具”转换为
“获得上层目标”的花费分布。常见组合已封装在 :doc:`basic_models`，无需重复实现。

数据约定与组合顺序
--------------------------------------

``GachaLayer.__call__`` 接收上一层的 ``(f, c)``，返回新一层的同类二元组：

* ``f`` 是从完整重置状态开始的一次目标花费分布。
* ``c`` 是从当前条件开始，获得首次目标的花费分布。
* 第一层输入 ``None``，直接生成抽数分布；后续层以底层成功次数作为自身计数单位。

当上层需要 ``k >= 1`` 个底层道具时，条件花费使用 ``c * f ** (k - 1)``。
因此层组合要求底层后续周期可独立重复，且上层规则能按这些成功次数建模。
完整分布通过层的默认条件计算，条件分布通过本次调用参数计算；
新增层应实现 ``_forward`` 并保留这两个分支的语义。

.. code-block:: python

   import numpy as np
   from GGanalysis.gacha_layers import PityLayer
   from GGanalysis.basic_models import DualPityModel

   # 第一层数抽数，第二层数第一层产出的道具数。
   bottom = PityLayer([0, 0.5, 1])(None, item_pity=1)
   full, first = PityLayer([0, 0.5, 1])(bottom, item_pity=0)
   expected = DualPityModel([0, 0.5, 1], [0, 0.5, 1])(item_pity=1)
   assert np.allclose(first.dist, expected.dist)

可用组件
--------------------------------------

* ``PityLayer``：接受条件成功概率表，或直接接受 ``FiniteDist`` 形式的成功次数分布。
  概率表与概率质量数组不能混淆：传入列表/数组时按概率表解释。
* ``BernoulliLayer``：每次底层成功以概率 ``p`` 成为目标，重复尝试直至成功。
* ``MarkovLayer``：通过列向量转移矩阵计算到达状态 0 的步数；默认从状态 0
  开始，至少执行一步，因此对应返回时间。``begin_pos`` 指定条件起点。
  需要保证能到达目标并检查截断；复杂命中边界优先使用 :doc:`hit_process`。
* ``CouponCollectorLayer``：每次底层成功等概率获得一种物品，以已有种类数
  ``initial_types`` 和目标种类数 ``target_types`` 描述一次集齐过程。
* ``DynamicProgrammingLayer``：由函数生成成功次数分布。源码仍标为实验性，
  接口和条件参数行为尚未充分验证；不作为新模型的默认选择。

在 ``CommonGachaModel`` 中，``layers`` 按底层到上层排列；
``_build_parameter_list`` 返回与层数一致的 ``[[args, kwargs], ...]``，
将模型参数传给对应层。示例见 :doc:`../start_using/custom_gacha_model`。

.. important::

   ``MarkovLayer`` 只在该层内部保留状态，输出仍是一维分布。
   它不会自动将命中后的状态传到下一次目标计算；存在跨次依赖时应使用
   ``StateKernel``，而不是对该层输出反复卷积。

API
--------------------------------------

``MarkovLayer`` 的首次返回递推复用公共 ``first_hitting_time``。
输入可以是普通守恒矩阵或 ``MarkovTransition``，编号 0 为目标及完整周期起点。
``p_error`` 对应剩余质量阈值，``max_steps`` 默认 10000 是计算上限；
达到上限仍未命中的质量保留在 ``FiniteDist.tail_mass``，不强制归一化。
它仍是独立周期的抽卡层适配器，不代替保留跨次状态的 ``HitProcess``。

.. autoclass:: GGanalysis.gacha_layers.GachaLayer
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.gacha_layers.PityLayer
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.gacha_layers.BernoulliLayer
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.gacha_layers.MarkovLayer
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.gacha_layers.CouponCollectorLayer
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.gacha_layers.DynamicProgrammingLayer
   :members:
   :special-members: __call__
