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

基础输入检查集中在 ``transition.py``，分析和命中过程复用同一实现。
首次命中允许非归一化质量；稳态幂迭代会归一化正质量初始权重；
事件计数、奖励和最终吸收分析要求归一化初态，不自动修复输入。

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

``tail_tol=None`` 为默认值，计算满指定步数；设置有限非负阈值时，每次吸收后
若剩余未命中质量不大于阈值就提前结束，返回数组截短至实际计算步数。
``include_t0=True`` 时也检查零步吸收后的剩余质量。``steps`` 仍是计算上限，
到达上限时可能尚未满足阈值，应检查返回的 ``surv``。不归一化或补全尾部；
尾部质量小不代表尾部期望、方差误差小。永不命中的质量超过阈值时不会提前停止。

.. code-block:: python

   import numpy as np
   from GGanalysis.markov import StateSpace, MarkovTransition, first_hitting_time

   space = StateSpace.from_shape([2])
   transition = MarkovTransition(space, np.array([[0.5, 1.0], [0.5, 0.0]]))
   f, surv = first_hitting_time(transition, space.delta([1]), [0], steps=2)
   assert np.allclose(f, [0, 1, 0])
   assert surv == 0

   f, surv = first_hitting_time(
       transition, space.delta([1]), [0], steps=100, tail_tol=1e-8,
   )
   assert np.allclose(f, [0, 1])
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
需要分解及初态相关结果时使用下述 ``ChainAnalysis``。

状态结构、最终吸收与累计奖励
--------------------------------------

``ChainAnalysis(transition)`` 保存转移和空间的快照，外部修改不影响分析。
``closed_classes`` 返回闭合互通类编号，``reachable_from(ids)`` 返回可达编号
并包含起点。结构按严格正概率边判断，不将极小概率边当作零。

``absorption(initial, targets)`` 接受归一化初态，以及名称到状态 selector 的映射。
各目标类别必须互不重叠；首次进入任何目标后停止，不要求原目标已经是吸收态。
返回 ``AbsorptionResult``，``probabilities`` 是各目标的竞争吸收概率，
``never_hit`` 是永不进入目标并集的概率。默认计入零步命中；
``include_t0=False`` 先传播一步，可计算首次返回。

内部排除无法到达目标的区域，提取继续状态上的 Q，解
``(I-Q) x = initial_U`` 得到期望访问次数，然后汇总进入目标的质量。
只缓存最近一次继续状态划分的线性分解，不构造逆矩阵。

.. code-block:: python

   import numpy as np
   from GGanalysis import ChainAnalysis, MarkovTransition, StateSpace, StateRewards

   # 从状态 0 下一步以 0.4 成功、0.6 失败，两个终态都吸收。
   chain = MarkovTransition(StateSpace.from_shape([3]), np.array([
       [0, 0, 0], [0.4, 1, 0], [0.6, 0, 1],
   ]))
   analysis = ChainAnalysis(chain)
   result = analysis.absorption([1, 0, 0], {"success": [1], "failure": [2]})
   assert result.probabilities == {"success": 0.4, "failure": 0.6}
   assert result.never_hit == 0
   rewards = StateRewards(chain.space, {"steps": [1, 1, 1]})
   total = analysis.expected_reward_until(
       [1, 0, 0], {"success": [1], "failure": [2]}, rewards,
   )
   assert total["steps"] == 1

``expected_reward_until`` 将访问次数与 ``StateRewards`` 的一步期望权重相乘。
包括进入目标的最后一步奖励，初始零步命中不产生奖励。要求从初态几乎必然
终止；存在非终止路径时拒绝计算，应明确增加失败终止目标，而不是隐式计算
成功条件期望。是否存在非终止路径按可达结构判断，不以小概率容差代替。

``long_run_state(initial)`` 分别求闭合类内部稳态，并按初态进入各类的概率混合。
返回长期平均占用比例；周期类的逐步分布可以振荡，不承诺逐步极限存在。

固定步数事件数量与奖励波动
--------------------------------------

``event_count_state_dist(event_matrices, initial, steps)`` 的第 r 个矩阵为
``K_r[end, start]``，表示一步获得 r 件及结束状态的联合概率。
矩阵总和须列归一化；一次可以获得多个，计数和转移不要求独立。
结果是 ``StateDist``，累计轴此时表示数量，通过 ``marginal_cost()`` 得到数量分布。

* ``strategy='step'`` 为默认，使用稀疏矩阵对活跃数量层批量顺序传播。
* ``strategy='power'`` 显式构造稠密 ``StateKernel``，再调用快速幂。
  大状态空间可能产生很大的中间核，不自动选择策略。
* ``method`` 控制 power 的 direct/fft/auto 卷积后端，不决定 step/power。

单步最多一件时，顺序传播时间为 O(steps**2 * E)，E 为非零转移边总数；
联合分布空间为 O(steps * N)。核快速幂减少组合次数，但核的数量轴随幂增长，
并非总计算量 O(log steps)，矩阵核自乘还可能破坏稀疏性。

.. code-block:: python

   import numpy as np
   from GGanalysis import event_count_state_dist, TransitionRewards

   events = [np.array([[0.7]]), np.array([[0.3]])]
   counts = event_count_state_dist(events, [1], 10).marginal_cost()
   reward = TransitionRewards.from_events(events)
   mean, variance = reward.cumulative_moments([1], 10)
   assert abs(mean - 3) < 1e-12
   assert abs(variance - 2.1) < 1e-12

``TransitionRewards`` 保存一个标量奖励的转移联合一阶矩 A 和二阶矩 B。
支持稀疏矩阵、随机及非整数奖励；不能仅将随机奖励均值平方当作二阶矩。
``from_events`` 用整数事件矩阵构造这两个矩。
``cumulative_moments`` 递推状态概率和累计奖励一、二阶矩，返回固定步数的
均值、方差，不构建完整数量分布。每步奖励条件分布不得额外依赖未编码的过去。

``ChainAnalysis.long_run_rewards(reward, initial)`` 要求奖励描述同一转移矩阵。
返回各闭合类的 ``class_weights``、``mean_rates``、``variance_rates``，以及混合
平均率 ``mean_rate`` 和类间平均率方差 ``between_class_variance``。
每类的 ``variance_rates`` 是累计奖励方差除以步数的极限，用泊松方程求解，
包括跨步相关性并允许周期类。若类间平均率方差非零，整体方差有平方级主项，
不能把各类线性方差增长率混合后当作整体线性增长率。

开发验证以几何周期、跨命中状态交替、一次多件、周期链和可约链作确定性对照。
状态周期法退化到 IID 工具时一致；事件分布的均值方差与矩递推一致；
独立 0/1 奖励的长期方差率为 p(1-p)，相关两状态奖励的结果与解析协方差和一致。

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
