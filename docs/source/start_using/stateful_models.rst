保留跨次状态的模型
======================================

普通分布卷积要求各阶段花费独立。若获得目标后状态没有完全重置，即使单次
花费的边缘分布相同，也不能据此重复卷积。本节用一个可手算的两状态过程说明区别。

例子：两种模式交替
--------------------------------------

假设过程有 A、B 两个模式：处于 A 时，下一个目标恰好花费 1 抽，随后进入 B；
处于 B 时，下一个目标恰好花费 2 抽，随后进入 A。
初始模式各占一半。

单次花费分布是 ``[0, 0.5, 0.5]``，但连续两次必定花费 3 抽。
若错误地对单次分布平方，会得到花费 2、3、4 抽分别为 0.25、0.5、0.25。

.. code-block:: python

   import numpy as np
   from GGanalysis.state_distribution import StateDist, StateKernel

   # coeff[花费, 结束模式, 起始模式]；A 编号 0，B 编号 1。
   coeff = np.zeros((3, 2, 2))
   coeff[1, 1, 0] = 1.0
   coeff[2, 0, 1] = 1.0
   kernel = StateKernel(coeff)
   kernel.validate()

   initial = StateDist([[0.5, 0.5]])  # 零花费，初始模式均匀混合
   first = kernel @ initial
   second = kernel @ first
   assert np.allclose(first.marginal_cost().dist, [0, 0.5, 0.5])
   assert np.allclose(second.marginal_cost().dist, [0, 0, 0, 1])
   assert np.allclose((first.marginal_cost() ** 2).dist, [0, 0, 0.25, 0.5, 0.25])

这个例子中，保留状态后下一轮规则完全确定。一般模型中，状态应包含影响未来的
全部信息，例如保底进度、特殊计数器或棋盘位置。
首次起点与后续命中边界不同的情况，可以先计算首次 ``StateDist``，再推进后续核。

连续目标与阶段预算
--------------------------------------

以下代码延续上例。``apply_power`` 重复应用规则，而不是假设各轮花费 IID。
若每个阶段都必须在累计预算内达成目标，应在阶段末剔除超预算质量，保留剩余状态。

.. code-block:: python

   third = kernel.apply_power(3, initial)
   assert np.allclose(third.marginal_cost().dist, [0, 0, 0, 0, 0.5, 0.5])

   joint = initial
   for cumulative_budget in [1, 3]:
       joint = kernel @ joint
       # 保留“花费、状态”联合分布；这里有意保留不足 1 的质量。
       joint = StateDist(joint.coeff[:cumulative_budget + 1])
   success_probability = joint.total_mass()
   assert success_probability == 0.5
   conditional_cost = joint.marginal_cost().normalized()
   assert conditional_cost.exp == 3

不要在每个阶段自动归一化，否则会丢失计划成功概率。
只有需要“成功玩家的条件花费分布”时，才对最后保留的质量归一化。
这里截去的质量代表计划失败；它与数值计算中因花费上限不足而漏掉的尾部是不同概念。

复用逐抽过程
--------------------------------------

如果已有逐抽转移规则，优先用 :doc:`../reference_manual/hit_process` 构建命中过程，
自动得到首次分布和周期核；不必手工推导每个 ``coeff[t, end, start]``。
若只需单次首次命中或普通稳态，可直接使用 :doc:`../reference_manual/markov`。
各类型和运算的定义见 :doc:`../reference_manual/state_distribution`。
