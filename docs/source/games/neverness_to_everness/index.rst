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

API
--------------------------------------

.. autofunction:: GGanalysis.games.neverness_to_everness.gacha_kernel.build_cycle_kernel

.. autofunction:: GGanalysis.games.neverness_to_everness.gacha_kernel.build_nte_analysis

.. autoclass:: GGanalysis.games.neverness_to_everness.gacha_model.NTEMonopolyModel
   :members: __call__, stationary_probability

.. autoclass:: GGanalysis.games.neverness_to_everness.stationary_statistic.NTEStationaryAnalysis
   :members:
