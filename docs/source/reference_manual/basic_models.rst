基础抽卡模型
======================================

定义位置：``GGanalysis/basic_models.py``。先使用现成模型，不能表达机制时再
组合 :doc:`gacha_layers`，或使用 :doc:`state_distribution` 保留跨次状态。

选择现成模型
--------------------------------------

.. list-table:: 模型与机制
   :header-rows: 1
   :widths: 42 58

   * - 模型
     - 表达的机制
   * - ``BernoulliGachaModel``
     - 每抽以固定概率成功，计算获得指定数量目标所需抽数
   * - ``PityModel``
     - 成功概率仅由当前保底进度决定，获得目标后重置
   * - ``DualPityModel``
     - 两层保底：底层获得道具，上层按底层道具次数保底获得目标
   * - ``PityBernoulliModel``
     - 底层保底，每次底层成功后以固定概率成为目标
   * - ``DualPityBernoulliModel``
     - 两层保底后，再以固定概率筛选目标
   * - ``CouponCollectorModel``
     - 每次等概率获得一种物品，计算集齐指定种类数所需次数
   * - ``PityCouponCollectorModel``、``DualPityCouponCollectorModel``
     - 一层或两层保底后等概率获得一种物品，计算集齐所需抽数
   * - ``GeneralCouponCollectorModel``
     - 每抽各物品概率可不同，按物品名称指定已拥有和目标集合

例如保底表 ``[0, 0.5, 1]`` 表示第一抽成功概率为 0.5，失败后第二抽必定成功。
第二层表 ``[0, 0.5, 1]`` 的单位则是底层道具个数，不是抽数。

.. code-block:: python

   from GGanalysis.basic_models import DualPityModel, GeneralCouponCollectorModel

   model = DualPityModel([0, 0.5, 1], [0, 0.5, 1])
   result = model(item_num=2, item_pity=1, up_pity=0)
   result.validate()
   assert abs(result.exp - 4.0) < 1e-10

   collection = GeneralCouponCollectorModel([0.2, 0.3], item_name=["A", "B"])
   # 剩余 0.5 概率不获得这两种物品；已拥有 A，只需等到 B。
   result = collection(init_item=["A"], target_item=["A", "B"])
   assert abs(result.exp - 1 / 0.3) < 1e-8

调用约定与边界
--------------------------------------

``GachaModel`` 是空基类，只提供类型层级，不强制子类签名。
``CommonGachaModel`` 提供抽卡层组合和多目标计算，其典型参数为：

* ``item_num``：需要获得的目标个数。
* ``multi_dist``：为真且目标数大于零时，返回从索引 0 到 ``item_num`` 的
  累计花费分布列表；索引 0 是零花费分布。
* ``item_pity``：当前已经连续未获得底层道具的抽数，不是下一抽的编号。
* ``up_pity``：上层保底进度，通常以底层道具次数计，不应当作抽数。

当前 ``CommonGachaModel(item_num=0, multi_dist=True)`` 仍直接返回
``FiniteDist([1])``，不是单元素列表。调用者应单独处理零目标。
集齐模型使用 ``initial_types`` / ``target_types`` 或 ``init_item`` / ``target_item``，
不套用普通多目标参数。等概率集齐的 ``target_types`` 表示累计拥有多少种，包含已拥有种类。
``BernoulliGachaModel`` 使用 ``calc_pull`` 指定计算上限，不提供 ``multi_dist``。

设首次条件花费分布为 ``c``，完全重置后的花费分布为 ``f``，则
``CommonGachaModel`` 对正整数 ``n`` 返回 ``c * f ** (n - 1)``。
这要求首次花费与后续周期独立，后续周期之间独立同分布。
若获得目标后仍继承影响未来的状态，不能直接套用此组合，见
:doc:`../start_using/stateful_models`。

自动截断模型的 ``e_error`` 通常控制期望相对误差，不能当作每个概率点的误差上限。
``max_dist_len`` 是迭代停止阈值，倍增搜索可能超过该值。
``BernoulliGachaModel`` 自动长度模式会记录理论期望、方差和尾部质量；显式
``calc_pull`` 模式不附带这些理论统计量。截断对象的统计解释见 :doc:`basic_tools`。

API
--------------------------------------

.. autoclass:: GGanalysis.basic_models.GachaModel
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.basic_models.CommonGachaModel
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.basic_models.BernoulliGachaModel
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.basic_models.PityModel
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.basic_models.DualPityModel
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.basic_models.PityBernoulliModel
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.basic_models.DualPityBernoulliModel
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.basic_models.CouponCollectorModel
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.basic_models.PityCouponCollectorModel
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.basic_models.DualPityCouponCollectorModel
   :members:
   :special-members: __call__

.. autoclass:: GGanalysis.basic_models.GeneralCouponCollectorModel
   :members:
   :special-members: __call__
