马尔可夫状态工具
======================================

定义位置：``GGanalysis/markov/``，公共入口为 ``GGanalysis.markov``。
普通转移记录每一步的状态；按“获得一次目标”组合花费则见 :doc:`hit_process`。

状态编码与转移方向
--------------------------------------

``StateSpace([2, 1])`` 的每维参数是包含上界的最大值，状态形状为 ``(3, 2)``。
等价写法是 ``StateSpace.from_shape([3, 2])``。编码按 C 顺序，最后一维变化最快。
``oob`` 默认对越界报错；只有模型确实需要截断或循环时才用 ``clip`` / ``wrap``。

状态选择器按每个维度填写整数、``None``、切片或编号集合。例如
``[None, 1]`` 选择第二维为 1 的所有状态。
``select_ids`` 返回编号，``hit`` 求选中质量，``clear`` 和 ``hit_and_clear`` 会原地修改概率向量。

统一采用 ``P[to, from]``、``p_next = P @ p``。``TransitionBuilder.add``
接收的参数顺序却是 ``(from_id, to_id, probability)``，不要倒置。
``add_state`` 可直接接受多维状态；重复边概率相加。
``build(check=True)`` 或 ``transition.validate()`` 检查完整转移的列概率和。
检查不修改矩阵；已移除 ``check_and_fix()``，不再提供自动修复。

矩形概率矩阵与快照
--------------------------------------

``ProbabilityMatrixBuilder(from_space, to_space, backend="sparse")`` 收集概率边，
返回形状为 ``(to_space.N, from_space.N)`` 的 CSR 矩阵；``backend="dense"``
返回 NumPy 数组。两端状态分别编码，允许不同维度和不同状态数。
它检查有限非负概率及整数编号，重复边累加，但不要求列和为 1。
``TransitionBuilder`` 复用它，负责同一空间上 ``MarkovTransition`` 的包装。

构建器配置在初始化后固定，边可以继续添加。添加非零边会使内部缓存失效；
``build()`` 和 ``matrix`` 均返回独立快照，修改结果不会改变构建器或此前的结果。
``MarkovTransition`` 本身仍可修改；三个稳态求解入口会重新检查转移概率守恒。

.. code-block:: python

   from GGanalysis.markov import ProbabilityMatrixBuilder, StateSpace

   full = StateSpace.from_shape([3])
   boundary = StateSpace.from_shape([2])
   edges = ProbabilityMatrixBuilder(full, boundary)
   edges.add_many([0, 0, 2], [1, 1, 0], [0.2, 0.3, 1.0])
   hit = edges.build()
   assert hit.shape == (2, 3)
   assert hit[1, 0] == 0.5

.. code-block:: python

   import numpy as np
   from GGanalysis.markov import StateSpace, TransitionBuilder, stationary_solve

   space = StateSpace.from_shape([2])
   builder = TransitionBuilder(space)
   builder.add_state([0], [0], 0.5)
   builder.add_state([0], [1], 0.5)
   builder.add_state([1], [0], 1.0)
   transition = builder.build(check=True)
   assert np.allclose(transition @ space.delta([0]), [0.5, 0.5])
   stationary, info = stationary_solve(transition, return_info=True)
   assert info.converged
   assert np.allclose(stationary, [2 / 3, 1 / 3])

首次命中与稳态
--------------------------------------

``first_hitting_time`` 对命中状态进行概率吸收，返回 ``(f, surv)``：
``f[t]`` 是第 t 步首次命中的质量，``surv`` 是计算结束仍未命中的质量。
它返回数组而非 ``FiniteDist``。``include_t0=True`` 将初始命中计入零步；
为假时先执行一步，可用于计算从目标状态出发的返回时间。
``return_pos=True`` 额外返回每个命中时刻的条件结束位置分布。

.. code-block:: python

   import numpy as np
   from GGanalysis.markov import StateSpace, MarkovTransition, first_hitting_time

   space = StateSpace.from_shape([2])
   transition = MarkovTransition(space, np.array([[0.5, 1.0], [0.5, 0.0]]))
   f, surv = first_hitting_time(transition, space.delta([1]), [0], steps=2)
   assert np.allclose(f, [0, 1, 0])
   assert surv == 0

稳态满足 ``P @ pi = pi``，不能用任意初态下的一次等待时间直接取倒数替代稳态概率。

.. list-table:: 稳态求解选择
   :header-rows: 1

   * - 方法
     - 适用条件与检查
   * - ``stationary_solve``
     - 解约束线性系统；``auto`` 在直接法和迭代法之间选择，查看 ``return_info=True`` 的残差
   * - ``stationary_power``
     - 逐步迭代；周期链可能振荡，可用 ``lazy`` 保持原稳态并抑制周期性，检查收敛标记
   * - ``stationary_eigs``
     - 求特征值 1 的向量；检查归一化和残差，不将选出的一个向量视为所有初态的长期结果

可约链可能存在多个稳态，线性系统可能奇异；初态决定进入各闭合类的概率。
这些接口不自动完成闭合类分解。``StationaryInfo`` 提供方法、是否收敛、迭代次数和残差。

事件、奖励与分类汇总
--------------------------------------

``analysis.py`` 中的 ``StateRewards`` 记录从各状态出发执行一步时的期望奖励。
输入是 ``{名称: 长度为 N 的权重向量}``；``expectation(p)`` 对归一化分布执行
``R @ p``，返回各名称对应的一步期望。传入逐抽稳态分布即得到长期每抽奖励率。
事件指示量的期望是事件概率，奖励数量可以大于 1，各事件也可以重叠。
因此奖励权重不是转移矩阵，不要求列和为 1，也不自动归一化。

.. code-block:: python

   from GGanalysis.markov import StateRewards, group_mass

   rewards = StateRewards(space, {
       "event": [0.5, 1.0],
       "bonus": [0.5, 0.0],  # 可以和 event 同时发生
       "coins": [2.0, 5.0],
   })
   rates = rewards.expectation([2 / 3, 1 / 3])
   assert abs(rates["event"] - 2 / 3) < 1e-12
   assert abs(rates["coins"] - 3.0) < 1e-12
   assert group_mass([0.2, 0.3, 0.1], ["A", "A", "B"]) == {"A": 0.5, "B": 0.1}

``group_mass(mass, labels)`` 只按同形状标签求和，不要求总质量为 1。
重叠事件用独立权重或掩码表达，不塞入互斥分类。条件统计由调用者显式选取分母，
例如“实际掷骰时的格子概率 = 每抽进入该格子的概率 / 每抽掷骰概率”；
必须确认分子属于条件事件，并处理零分母。

奖励权重由机制定义，尤其是重置之前发生的奖励，不能仅从合并后的转移矩阵推断。
这些工具不求稳态、不模拟、不补全截断尾部。权重与公开结果采用副本隔离。

多稀有度保底
--------------------------------------

``PriorityPityChain`` 按编号从小到大处理互斥道具，实际概率为该类原始概率与
尚未分配概率的较小值。``remove_pity=True`` 使高优先级命中也重置低优先级计数。
``stationary_rates`` 返回各类长期每抽概率；``interarrival_dist`` 是平稳条件下
相邻两次该类命中的间隔，其他类别在期间仍照常改变状态。

``stationary_item_count_distribution`` 只处理单类道具，初始保底进度按长期稳态混合；
不等于“从零垫抽开始”的数量分布。应用见 :doc:`../start_using/stationary_distribution`。

API
--------------------------------------

.. autoclass:: GGanalysis.markov.state_space.StateSpace
   :members:

.. autoclass:: GGanalysis.markov.transition.MarkovTransition
   :members:

.. autoclass:: GGanalysis.markov.builder.TransitionBuilder
   :members:

.. autoclass:: GGanalysis.markov.builder.ProbabilityMatrixBuilder
   :members:

.. automodule:: GGanalysis.markov.analysis
   :members:

.. autoclass:: GGanalysis.markov.priority_pity.PriorityPityChain
   :members:

.. autofunction:: GGanalysis.markov.priority_pity.stationary_item_count_distribution
