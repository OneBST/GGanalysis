异环
========================

本模型使用仓库中保存的测试服棋盘规则与数据，不代表自动同步的最新游戏规则。
规则定义见 ``gacha_data.py``，位置移动规则位于 ``gacha_kernel.py``。

角色棋盘模型
--------------------------------------

完整状态为 ``(连续未获得 S 的抽数, 棋盘位置)``，边界状态只保留位置。
获得 S 后保底计数归零、棋盘位置继承，因此相邻命中周期一般不独立同分布。
当前参数为 73 个位置、90 抽硬保底；连续未命中 70 抽后切换变格棋盘。
硬保底抽原地获得 S，不掷骰、不再次领取落点奖励。花费单位为一次主抽取，
不在花费中扣除格子奖励的骰子或兑换资源；附加事件的掷骰也不计作主抽取。

.. code-block:: python

   from GGanalysis.games.neverness_to_everness import up_5star_character

   first = up_5star_character(item_num=1, start_pos=0, item_pity=0)
   second = up_5star_character(item_num=2, start_pos=16, item_pity=25)
   kernel = up_5star_character.kernel
   kernel.validate(atol=1e-11)
   assert kernel.coeff.shape == (91, 73, 73)

   # 固定抽数下的 S 数量：首件条件分布加位置继承周期，不使用 IID 卷积。
   counts = up_5star_character.item_num_dist(100, start_pos=16, item_pity=25)
   # 多件花费默认顺序应用；快速幂须显式选择。
   third = up_5star_character(3, strategy="power", method="fft")

数量查询的 multi_dist 遍历投入抽数，多件花费查询的 multi_dist 遍历目标件数。
两个查询都保留位置依赖；只有输出时才汇总边界状态。

构建与复用
--------------------------------------

``gacha_kernel.py`` 定义游戏移动规则并生成一次掷骰的位置转移矩阵。
命中概率取决于落点，因此对该矩阵按行乘以概率，得到普通、变格阶段各自的命中与
未命中矩阵。硬保底层单独使用单位命中矩阵和零未命中矩阵。

随后调用 :meth:`GGanalysis.markov.hit_process.HitProcess.from_layers`，显式指定两个状态空间和
重置到第零层的 ``embed``。完整编号偏移、概率守恒检查、分层传播以及周期核生成
均由公共 Markov 工具负责。周期核只记录 73 个边界状态，不构造 6570 状态的稠密核。

``build_nte_analysis`` 是唯一的配置分析入口。同一组概率图及切换阈值共享一个
``HitProcessAnalysis(process, max_steps)``，不绑定初始位置或垫抽。
``build_cycle_kernel``、``build_nte_transition`` 与模型、稳态统计使用同一份内部缓存，
公开矩阵及核均返回独立副本，不再保证多次调用返回同一个对象。
棋盘移动规则集中在矩阵构建函数中，通过 ``ProbabilityMatrixBuilder.add_rule``
自动遍历状态和合并边，不再单独缓存逐位置的掷骰结果。

格子奖励标签、沉眠地追逐事件属于游戏规则，继续留在 ``stationary_statistic.py``。
该模块声明 ``StateRewards`` 的事件权重，由公共 ``expectation`` 计算每抽事件率；
``group_mass`` 按基础格子类型汇总，游戏显式除以实际掷骰率得到条件概率。
``C`` 与普通奖励重叠，单独保留；沉眠地统计复用已计算的 ``C`` 触发率。
模拟器保留独立的逐抽实现，便于交叉验证，不改成采样解析转移矩阵。

本次重构以旧实现保存的 16 组矩阵、周期核、不同垫抽首件分布、三件分布及稳态数组
做确定性比较，周期核系数逐项一致；另检查随机重置的局部层组装与逐边构建一致。
这些验证脚本保存在被忽略的 ``test/`` 目录。

作为新模型模板
--------------------------------------

构造层级和职责如下，箭头表示构建或使用关系::

   游戏数据与移动规则（gacha_data.py / gacha_kernel.py）
       │
       ├─ StateSpace：完整状态、边界状态及编码
       └─ ProbabilityMatrixBuilder：规则遍历、边累加、移动矩阵
               │ 游戏声明每层命中/未命中矩阵与 embed
               ▼
       HitProcess.from_layers：组装、验证、逐抽与首次命中递推
               │
               ▼
       HitProcessAnalysis：共享周期核和稳态，不绑定初态
               ├─ 初态参数 → first_state_dist / nth_state_dist / iter_state_dists
               │               └─ StateDist / StateKernel → FiniteDist
               └─ steady_full_state
                       │
   游戏声明事件权重 ── StateRewards.expectation：事件率、奖励期望
                       │
                       └─ group_mass：分类汇总
                               └─ NTEStationaryAnalysis：游戏名称、条件统计、追逐规则

``NTEMonopolyModel`` 是用户调用入口，负责解释位置、垫抽和目标数量等游戏参数，
多件递推交给 ``HitProcessAnalysis``，再将联合分布转换为 ``FiniteDist``。
公共文件职责：``builder.py`` 负责构建；``transition.py`` 负责转移与配套验证；
``hit_process.py`` 负责命中过程及分析；``analysis.py`` 负责稳态求解、奖励期望和分类汇总。
不为奖励统计另建模块。

.. code-block:: python

   import numpy as np
   from GGanalysis.games.neverness_to_everness import (
       build_nte_analysis, character_stationary,
       CHARACTER_POOL_NORMAL_MAP_5, CHARACTER_POOL_PITY_MAP_5, PITY_START,
   )

   analysis = build_nte_analysis(
       CHARACTER_POOL_NORMAL_MAP_5, CHARACTER_POOL_PITY_MAP_5, PITY_START,
   )
   positions = analysis.process.boundary_space
   # 改初态只改变方法参数，周期核和长期统计继续共享。
   a = analysis.first_state_dist(positions.delta([0]), initial_layer=0)
   b = analysis.nth_state_dist(2, positions.delta([16]), initial_layer=25)
   assert abs(a.total_mass() - 1) < 1e-11
   assert abs(b.total_mass() - 1) < 1e-11
   rates = character_stationary.event_rewards.expectation(analysis.steady_full_state)
   assert np.isclose(rates["C"], character_stationary.nacupeda_statistics.trigger_probability_per_pull)

新游戏只需替换规则数据、状态空间、命中判定与重置、事件权重。不要复制周期卷积、
稳态求解或通用分类求和；也不要让公共模块理解棋盘格字符串。

长期统计
--------------------------------------

``model.stationary`` 提供命中后位置分布、逐抽位置和保底分布、格子进入率及沉眠地统计。
长期 S 概率由公共分析的稳态周期期望倒数给出。零垫抽的首件分布与稳态周期分布不同，
不可互换；格子进入率还需排除不掷骰的硬保底抽。

角色棋盘的集点赠礼
--------------------------------------

按集点赠礼规则，每累计 10 次主抽额外赠送一件 A 级道具，20% 为 A 角色、
80% 为 A 弧盘。该赠礼不替换落格奖励，自然获得 A 或 S 不重置赠礼进度；
原地 S 硬保底也计入主抽次数。限定棋盘之间共享赠礼计数。
因此角色棋盘没有用于该赠礼的“连续未出 A”计数，不采用武器池的 A 保底转移。

.. code-block:: python

   from GGanalysis.games.neverness_to_everness import character_stationary

   rates = character_stationary.a_grade_stationary_rates
   # 接下来抽 21 次，已有 9 点赠礼进度：固定获得三次额外赠礼。
   gift = character_stationary.point_gift_dist(pull=21, gift_pity=9)
   assert abs(gift["A_character"].exp - 0.6) < 1e-12
   assert abs(gift["A_arc_disc"].exp - 2.4) < 1e-12

``point_gift_dist`` 只含额外赠礼，返回两类数量的边缘分布；二者之和是固定赠礼总数，
不能按独立分布卷积。单次赠礼类型按独立的 20%/80% 抽样建模。
``a_grade_stationary_rates`` 将 A/W 落格率分别加上 0.02/0.08，得到每主抽的
A 角色/A 弧盘期望数量。当前保存的棋盘计算结果约为 11.6749446731%、
11.3156000590%，合计 22.9905447322%。这是含额外奖励的等效产率，
同抽允许获得两件 A，不是“至少得到一个 A”的事件概率。
原 ``tile_entry_probabilities`` 仍只记录棋盘落格事件。

公示文案为 11.6700%/11.3100%，合计 22.9800%；当前模型不通过修改移动概率
或强制归一化拟合该数值。规则来源为用户提供的集点赠礼条款，公开资料可参考
`卡池规则转载 <https://kater.me/c/7v9wd2rmnj?bid=73>`_；
`官方限定棋盘公告 <https://steamcommunity.com/app/4508340/allnews/?l=schinese>`_
亦明确限定棋盘共享集点赠礼计数。

装扮首次获得与长期重访
--------------------------------------

本仓库位置编号以起点 0 为基准；普通和变格限定棋盘上的风向标 ``HD3`` 均在
位置 3，涂装 ``HD2`` 在位置 56，时装 ``HD1`` 在位置 65。
分别在同名棋盘累计 50/120/200 次主抽可领取一次装扮礼遇，不因提前落格获得而
重置累计进度，也不作为每次获得后反复重置的保底。同名棋盘累计进度继承；
两类首获查询均假设尚未拥有目标装扮、装扮累计从零开始，并在到点时立即领取。
零初态取位置 0、S 进度 0；平稳起态取长期刚出 S 后的位置稳态、S 进度 0。
后者不是随机主抽时点的位置与 S 进度联合稳态。
重复时装/涂装/滑翔翼分别兑换 16/16/4 失纬棋子，不改变棋盘运动。

设 P 为完整角色过程转移，J 为实际进入目标装扮格的转移，K=P-J。
J 必须排除原地 S 硬保底；停留在目标格并不等于再次领取格子奖励。
正式实现位于 ``cosmetic_model.py``，以 ``NTECosmeticModel`` 组合已有角色转移、
``first_hitting_time`` 和 ``FiniteDist``，不新增公共 Markov 算法。
将 J 汇入额外吸收态，只计算剩余领取期限 C 之前的首达概率；最后仍未获得的
路径全部在 C 领取，得到 min(自然首达时间, C) 的完整分布，tail_mass 为零。
这不是无限尾部分布的截断，不归一化、不递推至 2000 抽。
计算过程保留 S 进度、位置及出 S 后的位置继承。

.. code-block:: python

   from GGanalysis.games.neverness_to_everness import up_5star_character

   wing = up_5star_character.cosmetics["wing"]  # 另有 paint/outfit
   initial = wing()                           # 位置0、S进度0、装扮累计0
   stationary = wing.stationary_start_dist()  # 长期刚出S后的位置稳态，S/装扮进度0
   conditional = wing(start_pos=16, item_pity=25, cosmetic_pity=10)
   print(initial.exp, stationary.exp)         # 两种有限首获分布的期望
   print(wing.stationary_probability)         # 长期每主抽实际落格概率
   print(wing.revisit_expectation)            # 直接取上述概率的倒数
   # 如需表达某一起态的首获“等效综合率”，可取期望倒数，不能当作长期落格率。
   print(1 / stationary.exp)

.. list-table:: 从零初态出发的期望与长期实际落格重访（主抽）
   :header-rows: 1

   * - 装扮
     - 位置
     - 含累计领取首获均值
     - 平稳位置起态含领取首获均值
     - 长期落格重访均值
   * - 滑翔翼
     - 3
     - 29.1775698399
     - 32.6063786601
     - 58.4396568852
   * - 车辆涂装
     - 56
     - 97.4666386588
     - 96.8561827960
     - 301.2871610332
   * - 角色时装
     - 65
     - 147.4996607732
     - 145.3324467394
     - 301.2871610332

长期均值直接取目标实际进入率的倒数；一次性的累计领取不贡献无限长期产率。
模型只提供长期落格概率与平均重访，不生成无保底的重访分布。
两类首获分布均与独立的按 S 进度层递推核验，概率完整；另检查领取期限、
已达领取次数、原地 S 硬保底不领格子奖励但照常累计领取进度等边界。

公示“首次综合概率”不能直接作为独立伯努利概率使用，其起态口径未由这些条款
唯一确定。上表明确区分两种起态，不强制使其期望倒数等于公示综合概率。
验证脚本及首获分布 CSV 保留在被忽略的 ``test/``。

弧盘研募：UP S 与综合概率
--------------------------------------

武器池采用本地分析目录 ``2026-10-24-异环抽卡解析视频/analysis_code/weapon.py``
及配套规则台账选用的逐件口径，不自动同步服务端机制。每件基础 S 概率 3%，
其中 UP 占 25%，基础 A 概率 7%；60 件保 S、80 件保 UP、10 件保 A。
优先级为 UP 保底、S 保底、A 保底。A级保底不挤占 S，剩余概率全部给 A。

UP 同时清 S/UP 计数，非 UP S 只清 S。所有 S 均不清 A 计数；实际 A 才清 A，
未出 A 的计数在 9 饱和，遇到 S 优先时继续等待 A。计数跨期继承。
综合概率使用抽取前状态 ``(连续未出 S 件数, 连续未出 UP 件数, 连续未出 A 件数)``，
从零状态枚举 29555 个可达状态，构造稀疏转移，再通过公共稳态求解和
``StateRewards`` 加权计算长期每件率，不使用模拟。

.. code-block:: python

   from GGanalysis.games.neverness_to_everness import (
       up_5star_arc_disc, arc_disc_stationary_rates,
   )

   first = up_5star_arc_disc()
   five = up_5star_arc_disc(item_num=5)
   conditional = up_5star_arc_disc(item_pity=20, up_pity=70)
   rates = arc_disc_stationary_rates()
   # S 包含 UP；互斥品质为 S/A/B，不能再加一次 UP。
   assert abs(rates["S"] + rates["A"] + rates["B"] - 1) < 1e-11
   assert abs(rates["UP"] - 1 / first.exp) < 1e-11

长期 S、UP、A、B 综合概率分别约为 4.1981424982%、1.6830092953%、
13.4752295660%、82.3266279358%。这些是每件长期率，不是指定预算的获得概率。
该实现与参考逐件规则在所有可达状态逐项核对，并检查概率守恒、稳态残差和
UP 更新周期率；1—5 把 UP 的花费分布另用独立数量递推核验。

``up_5star_arc_disc`` 只返回 UP S 的累计逐件花费分布。UP 后 S/UP 计数归零，
A 计数虽然继承，但不影响下次 S/UP 的概率，因此后续 UP 周期可以 IID 卷积。
零状态首 UP 均值约为 59.4173783123 件。不再提供 ``first_5star_arc_disc``
普通 S 等待分布入口；任意 S 目前只计算长期综合概率。

十连支付成本
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若只能以十连支付，取得第 k 把 UP 的总件数为 T，则钥匙花费为
``10 * ceil(T / 10)``。应先计算累计件数的分布，再整体取整；
不能将每把的等待分别取整后相加，否则会丢失同一十连里的多把 UP。
这是相同逐件过程的支付观察，不改变包内保底规则。

.. code-block:: python

   import numpy as np
   from GGanalysis.distribution_1d import FiniteDist

   total = up_5star_arc_disc(item_num=5)
   # 下标为支付的十连包数，乘 10 后为钥匙数。
   package_index = (np.arange(len(total.dist)) + 9) // 10
   paid_packages = FiniteDist(np.bincount(package_index, weights=total.dist))
   expected_keys = 10 * paid_packages.exp

零状态首 UP 的十连支付平均成本约为 61.5283479968 钥匙。

API
--------------------------------------

.. autofunction:: GGanalysis.games.neverness_to_everness.gacha_kernel.build_cycle_kernel

.. autofunction:: GGanalysis.games.neverness_to_everness.gacha_kernel.build_nte_analysis

.. autoclass:: GGanalysis.games.neverness_to_everness.gacha_model.NTEMonopolyModel
   :members: __call__, stationary_probability, cosmetics

.. autoclass:: GGanalysis.games.neverness_to_everness.cosmetic_model.NTECosmeticModel
   :members: __call__, stationary_start_dist, stationary_probability, revisit_expectation

.. autoclass:: GGanalysis.games.neverness_to_everness.stationary_statistic.NTEStationaryAnalysis
   :members:

.. autoclass:: GGanalysis.games.neverness_to_everness.gacha_model.NTEArcDiscUPModel
   :members: __call__

.. autofunction:: GGanalysis.games.neverness_to_everness.stationary_statistic.arc_disc_stationary_rates
