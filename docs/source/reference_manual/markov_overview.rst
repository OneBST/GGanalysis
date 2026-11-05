Markov 总览与工具选择
======================================

从这里开始了解 ``GGanalysis.markov``。本模块适合用有限离散状态描述的随机过程：
给定当前状态，就能确定下一步的转移和奖励规则。一步可以是一抽、一次五星结算
或一次掷骰，必须由模型明确；工具不会自动换算花费单位。

先选择模型，再选择算法
--------------------------------------

已有普通保底、大小保底模型时，先查 :doc:`basic_models`。若每次获得目标后
完全重置、后续等待周期 IID，就不必为了统一形式改写成状态链。
若目标获得后仍继承模式、位置或计数器，则需要保留这些状态。
``MarkovLayer`` 可将一次马尔可夫等待嵌入抽卡层，但不会自动保留跨次边界状态。
若外层用 ``CommonGachaModel`` 重复卷积，仍须满足后续周期 IID；否则改用
命中过程与状态核衔接，见 :doc:`gacha_layers` 和 :doc:`state_distribution`。

.. list-table:: 按问题选择入口
   :header-rows: 1
   :widths: 31 33 36

   * - 要回答的问题
     - 推荐工具
     - 先确认什么
   * - 某一步的状态分布
     - ``MarkovTransition``
     - 完整状态、单步单位及列向量方向
   * - 第一次进入目标需要几步
     - ``first_hitting_time``
     - 初始命中是否算零步，计算上限及剩余质量
   * - 最终能否命中、先进入哪个目标
     - ``ChainAnalysis.absorption``
     - 目标组互斥，结果允许存在永不命中概率
   * - 可达区域、闭合类、初态相关长期占用
     - ``ChainAnalysis``
     - 长期平均与逐步分布收敛不同
   * - 多次获得目标的累计花费
     - ``HitProcessAnalysis``
     - 命中后的边界状态必须足以决定下一轮
   * - 固定预算内数量，已有 IID 等待周期
     - ``independent_item_num_dist``
     - 正整数等待，每抽至多一件，首件可单独指定
   * - 固定预算内数量，已有状态等待周期
     - ``stateful_item_num_dist`` 或命中分析的 ``item_num_dist``
     - 首件和周期核覆盖预算内全部系数
   * - 固定步数内事件数量及最终内部状态
     - ``event_count_state_dist``
     - 单步事件矩阵 K_r；允许一步多件
   * - 只需要奖励均值、方差
     - ``StateRewards``、``TransitionRewards``
     - 一步期望与联合二阶矩不是同一种输入
   * - 多稀有度互相挤占、保底重置
     - ``PriorityPityChain``
     - 类别优先级、重置规则及观察时刻
   * - 每次独立抽取，集齐指定种类
     - ``GeneralCouponCollection``
     - 每种道具概率固定，每次至多一件，状态只需已拥有集合

结构与依赖方向
--------------------------------------

目录保持平铺；以下是职责分组，不是三组子目录。箭头表示“使用”：

.. code-block:: text

   具体模型：priority_pity、coupon_collection
       使用通用计算（coupon_collection 保留专用集合递推）
   通用计算：analysis、hit_process
       使用状态、转移和矩阵构造
   基础表达：state_space、transition、builder

.. list-table:: 各文件负责什么
   :header-rows: 1

   * - 文件
     - 职责和边界
   * - ``state_space.py``
     - 编码、选择状态；不知道某条边的概率或某个游戏规则
   * - ``transition.py``
     - 普通转移、共用检查和事件矩阵格式整理；不导入具体保底模型
   * - ``builder.py``
     - 按边收集普通/矩形概率矩阵；不推导机制
   * - ``analysis.py``
     - 首次命中、稳态、吸收、结构、事件数量和奖励
   * - ``hit_process.py``
     - 区分命中/未命中，明确重置，生成状态等待周期
   * - ``priority_pity.py``
     - 把优先级保底规则变成状态转移，复用分析工具
   * - ``coupon_collection.py``
     - 位掩码集合与专用集齐递推，不强制构造普通转移矩阵
   * - ``__init__.py``
     - 公开入口，不放机制或递推算法

``StateDist``、``StateKernel`` 在包外层的 ``state_distribution.py``，是通用
花费与状态类型，并非具体应用。分析与命中过程使用它们；基础转移工具
只接收矩阵及矩阵序列，不需要识别具体状态核类型。
游戏特有的计数器更新、概率表和重置规则留在游戏目录，高层复用底层，底层
不反向导入游戏或优先级保底模型。内部使用明确的实现模块导入；调用者通常
从 ``GGanalysis.markov`` 或根包导入同一对象，NumPy/SciPy 则直接从原库导入。

从机制构造一个可检查的模型
--------------------------------------

1. 明确目标事件、一步的单位、起点和停止条件。
2. 写出影响未来的完整状态；命中后可舍弃的信息与须继承的信息分开。
3. 选择普通转移构造，或需要反复命中时的命中/未命中构造。
4. 检查概率守恒，先用小模型核对一两步，再选择长期或多次查询。
5. 保留截断剩余质量，核对已知结果；不靠归一化掩盖遗漏。

普通过程有两种构造入口：已有完整矩阵用 ``MarkovTransition(space, P)``，
有机制边则用 ``TransitionBuilder``。不同起点/终点空间用
``ProbabilityMatrixBuilder``；它的矩形矩阵不一定列归一化，例如只记录命中分支。
命中过程可传完整 M、H、E 给 ``HitProcess``，也可通过
``HitTransitionBuilder`` 收集边；只有严格按连续未命中次数推进的机制才用
``HitProcess.from_layers``，详见 :doc:`hit_process`。

下面是独立可运行的普通过程例子：首抽一半成功，失败后下一抽必成，成功后归零。

.. code-block:: python

   import numpy as np
   from GGanalysis.markov import StateSpace, TransitionBuilder, first_hitting_time

   space = StateSpace.from_shape([2])
   builder = TransitionBuilder(space)
   builder.add(0, 0, 0.5)
   builder.add(0, 1, 0.5)
   builder.add(1, 0, 1.0)
   transition = builder.build(check=True)
   f, remaining = first_hitting_time(
       transition, space.delta([0]), [0], steps=2, include_t0=False,
   )
   assert np.allclose(f, [0, 0.5, 0.5])
   assert remaining == 0

状态编号和物理状态不同
--------------------------------------

``StateSpace`` 不要求按编号依次转移。多维整数状态可直接编码；不规则的有限
状态可由调用者维护“业务状态 ↔ 一维编号”映射，再用 ``from_shape([N])``。
随机性由边的概率表示，不要求给随机状态另建一种 StateSpace。

例如三个不规则模式可这样构造；编号 0、1、2 只是标签，不意味着顺序：

.. code-block:: python

   import numpy as np
   from GGanalysis.markov import StateSpace, TransitionBuilder

   ids = {"普通": 0, "保证": 1, "特殊": 2}
   space = StateSpace.from_shape([len(ids)])
   builder = TransitionBuilder(space)
   builder.add(ids["普通"], ids["普通"], 0.7)
   builder.add(ids["普通"], ids["特殊"], 0.3)
   builder.add(ids["保证"], ids["普通"], 1)
   builder.add(ids["特殊"], ids["保证"], 1)
   transition = builder.build(check=True)
   assert np.allclose(transition @ space.delta([ids["普通"]]), [0.7, 0, 0.3])

不要用 ``clip`` 随意合并高计数器状态；只有证明它们之后的行为等价时才能精确
合并。多维笛卡尔积过大时，优先枚举可达状态并编号，而不是先建巨大稠密矩阵。

两个累计轴不能混淆
--------------------------------------

``StateKernel`` 是三维系数容器，第一轴表示哪种可加的整数累计量，由机制决定：

.. list-table:: 两种核
   :header-rows: 1

   * - 含义
     - 第 r 层表示什么
     - 重复一次对应什么
   * - 等待花费核
     - 等待 r 抽获得下一件，并进入相应边界状态
     - 再获得一件
   * - 单步事件计数核
     - 执行一步获得 r 件，并进入相应完整状态
     - 再执行一步

分析入口先从 ``StateKernel`` 取出 coeff；基础工具 ``prepare_event_matrices``
只把矩阵序列逐层转为 CSR 并检查联合转移，
不会将等待花费核转换成单步计数核。单步矩阵 K_r 的和必须列归一化，
各 K_r 单独不要求列和为 1。``StateKernel`` 自身仍是稠密类型。

下面的两个核都来自单抽概率 0.3，但输入和累计量完全不同：

.. code-block:: python

   import numpy as np
   from GGanalysis import StateDist, StateKernel, stateful_item_num_dist
   from GGanalysis.markov import event_count_state_dist

   # 单步事件核：一抽获得 0 或 1 件，可保留稀疏矩阵，不必先建 StateKernel。
   events = [np.array([[0.7]]), np.array([[0.3]])]
   count_joint = event_count_state_dist(events, [1], steps=10)
   by_step = count_joint.marginal_cost()
   # 等待花费核：直到下一件，等待 1～10 抽；预算 10 以外系数无需计算。
   waiting = np.r_[0, 0.3 * 0.7**np.arange(10)]
   by_cycle = stateful_item_num_dist(
       StateDist(waiting[:, None]), StateKernel(waiting[:, None, None]), pull=10,
   )
   assert np.allclose(by_step.dist, by_cycle.dist)

如果还要知道预算结束时的完整状态，选逐步事件计数；周期数量方法只输出数量，
不能把最后一次命中的边界状态当作预算结束状态。只有均值方差时优先考虑矩递推，
避免构造完整数量分布。

计算策略和结果解释
--------------------------------------

* 默认先用 ``step`` 顺序推进。``power`` 会构造稠密中间核，适合状态较少且
  已确认内存可承受的重复过程；不因步数大就直接认定快速幂更快。
* ``method=direct/fft/auto`` 选择卷积后端，不选择 step/power，也不改变概率语义。
* ``max_steps``、``steps`` 是计算上限，``tail_tol`` 是首次命中的提前停止阈值；
  不是所有命中接口都支持 tail_tol。检查具体方法返回的剩余质量，不假定自动收敛。
* 原始稳态求解器适用于选定的稳态问题；可约链或初态相关结果用 ``ChainAnalysis``。
  长期平均占用即使存在，也不代表周期链逐步分布收敛。
* 吸收前累计奖励要求从初态几乎必然终止。若有失败分支，应将失败也明确列为
  终止目标；不能将成功条件期望与无条件期望混为一谈。
* ``StateRewards`` 保存状态条件的一步期望；``TransitionRewards`` 保存与下一状态
  联合的一、二阶矩。随机奖励的二阶矩一般不是均值平方。

继续阅读
--------------------------------------

* 普通转移、首次命中、最终吸收和奖励：:doc:`markov`。
* 命中/未命中、边界重置和完整构造示例：:doc:`hit_process`。
* 核组合、预算裁剪与概率质量：:doc:`state_distribution`。
* 集齐问题的适用条件和位掩码构造：:doc:`coupon_collection`。
* 多类别保底和两种观察时刻：:doc:`../start_using/stationary_distribution`。
* 跨次状态导致非 IID 的手算例子：:doc:`../start_using/stateful_models`。
