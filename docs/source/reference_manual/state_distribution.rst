状态分布与状态核
======================================

定义位置：``GGanalysis/state_distribution.py``。
当一次目标的花费与结束状态相关，而且结束状态影响下一阶段时，使用这些类型。
完整例子见 :doc:`../start_using/stateful_models`。

三个类型的区别
--------------------------------------

.. list-table:: 概率系数约定
   :header-rows: 1
   :widths: 22 33 45

   * - 类型
     - 索引
     - 含义
   * - ``FiniteDist``
     - ``dist[t]``
     - 花费恰为 t 的概率
   * - ``StateDist``
     - ``coeff[t, s]``
     - 已给定初始条件，花费为 t 且结束状态为 s 的联合概率
   * - ``StateKernel``
     - ``coeff[t, end, start]``
     - 从 start 出发，花费 t 后结束于 end 的联合条件概率

花费轴始终在第 0 轴，状态编号的含义由调用者维护。两端状态数相等不代表
状态含义一致，组合前必须确认编码相同。``coeff`` 返回只读视图，修改系数应复制后重建对象。

状态分布
--------------------------------------

``StateDist.delta`` 创建确定初态；``from_product`` 仅适用于花费与状态独立的情况。
``mixture`` 按权重叠加联合分布；调用者负责保证概率权重合法。

``marginal_cost()`` 对状态求和，``marginal_state()`` 对花费求和。
后者将状态编号作为 ``FiniteDist`` 的索引，编号本身通常没有可解释的数值期望。
``condition_state`` 和 ``condition_cost`` 会除以对应事件质量，返回条件概率；
条件事件质量为零时抛出异常。

状态核运算
--------------------------------------

* ``B @ A`` 或 ``B.compose(A)``：先 A 后 B，花费相加，对中间状态求和。
* ``K @ joint`` 或 ``K.apply(joint)``：向前推进已有联合分布。
* ``K ** n``：同一规则重复 n 次；必须为方形核，且前后状态语义一致。
* ``K.apply_power(n, joint)``：直接将重复过程作用于初始分布。
* ``K.cost_dist(start_state)``：给定起始状态的一次花费边缘分布。
* ``K.marginal_transition()``：对花费求和，得到普通列向量转移矩阵。

重复核不要求各轮花费 IID，但要求状态包含影响未来的全部信息。
``StateDist`` 没有两个分布间的卷积运算，因为它没有起始状态轴。
核的 ``*`` 是数乘；组合应使用 ``@``。

``identity`` 创建零花费恒等核，``from_finite_dist`` 将普通花费嵌入为不改变状态的核。
``deterministic(transition, cost)`` 指定确定花费；状态转移矩阵本身仍可以随机。
``compose``、``apply``、``apply_power`` 支持 ``method="auto"``、``"direct"``、``"fft"``。
直接算法适合小规模核及数值对照，FFT 适合较长花费轴；二者都存在浮点误差。

概率质量与截断
--------------------------------------

``StateDist.total_mass()`` 返回已表示的质量，``StateKernel.validate()``
检查每个起点对应的总质量是否为 1 及是否存在明显负系数。
构造器并不代替完整的概率合法性检查，输入也应检查有限值。
``StateDist`` 和 ``StateKernel`` 没有独立的尾质量字段，截断信息需由调用者记录。

若初始质量为 1，截断结果的 ``1 - joint.total_mass()`` 是未表示的质量；
对核应分别检查 ``1 - kernel.coeff.sum(axis=(0, 1))``，不能只看某一列。
不要自动归一化每列来通过验证，这会将原过程改为条件过程。

API
--------------------------------------

.. autoclass:: GGanalysis.state_distribution.StateDist
   :members:

.. autoclass:: GGanalysis.state_distribution.StateKernel
   :members:
