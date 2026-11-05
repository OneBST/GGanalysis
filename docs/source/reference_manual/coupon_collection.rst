固定概率的集齐问题
======================================

定义位置：``GGanalysis/markov/coupon_collection.py``。公开入口为
``GGanalysis.markov.GeneralCouponCollection``，也可从根包导入。
工具选择与其他 Markov 组件的关系见 :doc:`markov_overview`。

适用条件
--------------------------------------

每次独立抽取至多获得一种目标道具，各种概率固定，重复道具不改变后续规则。
状态只记录“已经拥有哪些种类”，不记录数量。概率和可以小于 1，剩余概率是
未获得任何目标种类；各目标概率应为有限正数，总和不超过 1。
当前构造器支持 1～15 个目标种类，状态数随种类数按 2**m 增长。

若有保底进度、不同模式、抽到某物后改变概率，不能直接输入平均概率代替这些
状态。优先查看现成集齐模型；任意状态依赖则用普通状态链或命中过程。
这里保留专用位掩码递推，不强行构造 2**m 大小的普通转移矩阵。

状态与构造
--------------------------------------

``p_list[i]`` 是每抽获得第 i 类道具的概率；``item_names`` 可选，与概率表等长。
状态整数的第 i 位为 1 表示拥有该类。例如 A、B、C 编号依次为 0、1、2，
状态 5（二进制 101）表示已有 A、C，状态 7 表示全部集齐。
默认从 0 开始，目标为全部位。提供名字后可通过 ``encode_state_number``、
``decode_state_number`` 在名字列表与位掩码间转换。
目标位掩码表达“至少拥有这些种类”，不要求没有额外种类。

下面的例子完整可运行：每抽 A、B 各占 1/4，另有 1/2 没有目标道具。

.. code-block:: python

   import numpy as np
   from GGanalysis import FiniteDist
   from GGanalysis.markov import GeneralCouponCollection

   model = GeneralCouponCollection([0.25, 0.25], item_names=["A", "B"])
   a = model.encode_state_number(["A"])
   both = model.encode_state_number(["A", "B"])
   assert a == 1 and both == 3
   assert model.decode_state_number(both) == ["A", "B"]
   assert model.get_expectation() == 6
   assert model.get_expectation(state=a, target_state=both) == 4
   cdf = model.get_collection_p(20)
   assert np.allclose(cdf[:3], [0, 0, 0.125])
   pmf = np.diff(np.r_[0, cdf])
   truncated = FiniteDist(pmf, tail_mass=1-cdf[-1])
   assert abs(truncated.p_sum + truncated.tail_mass - 1) < 1e-12
   # 已经拥有目标时，零抽即满足，而不是从下一抽才开始计入。
   assert model.get_collection_p(0, init_state=a, target_state=a)[0] == 1

结果和计算方式
--------------------------------------

* ``get_expectation(state, target_state)`` 是无限等待范围下的解析递归期望，
  不依赖下面的有限抽数截断。构造后不要修改概率表，否则缓存期望将失效。
* ``get_collection_p(n)`` 返回长度 n+1 的累计概率数组，索引 t 表示 t 抽内
  已满足目标的概率，不是恰好在第 t 抽集齐的概率。相邻差才是首次完成分布。
* ``iter_collection_p`` 是从零抽开始的无限迭代器，只保留当前集合分布；
  调用者必须自行停止，不直接将其转换成完整列表。
* ``collection_dp(n)`` 返回完整状态分布，形状为 (2**m,n+1)，不是首次命中分布。
  仅需要集齐概率时优先用前两种方法，避免保存所有历史状态。
* ``set_default_init_state``、``set_default_target_state`` 改变后续查询默认起点和
  目标；单次查询可直接传状态参数，避免在不同调用者之间共享可变默认设置。

下面独立示例说明如何复用完整 DP，以及等概率特例：

.. code-block:: python

   import numpy as np
   from GGanalysis.markov import GeneralCouponCollection, get_equal_coupon_collection_exp

   model = GeneralCouponCollection([0.5, 0.5])
   dp = model.collection_dp(5)
   cdf = model.get_collection_p(5, DP_array=dp)
   assert dp.shape == (4, 6)
   assert np.allclose(cdf, model.get_collection_p(5))
   # 两类等概率，每抽必得其中一类：从零集齐期望为 2*(1+1/2)=3。
   assert get_equal_coupon_collection_exp(2) == model.get_expectation() == 3

``get_equal_coupon_collection_exp`` 假设每抽必得 item_type 种中的一种且等概率，
init_type、target_type 是拥有种数，不是位掩码。它没有“未获得目标”的空抽分支。
有限 n 的 CDF 未达到 1 是真实尾部，不补到最后一项，也不归一化。
截断 ``FiniteDist`` 自动计算的矩不能当作完整期望；完整期望使用 get_expectation。
``sim_collection`` 为可选模拟接口，解析和 DP 可行时不作为默认计算手段。

API
--------------------------------------

.. autoclass:: GGanalysis.markov.coupon_collection.GeneralCouponCollection
   :members:

.. autofunction:: GGanalysis.markov.coupon_collection.get_equal_coupon_collection_exp
