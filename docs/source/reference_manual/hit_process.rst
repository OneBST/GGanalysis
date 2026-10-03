带命中标记的过程
======================================

定义位置：``GGanalysis/markov/hit_process.py``，从 ``GGanalysis.markov`` 导入。
适用于每步至多获得一个目标、规则不随时间改变的有限状态过程。
它将逐步转移转换成 :doc:`state_distribution`，避免每个游戏重复实现命中递推。

完整状态与边界状态
--------------------------------------

完整状态记录每次抽取前需要的全部信息；边界状态记录刚获得目标后，影响下一轮的全部信息。
例如完整状态为“保底进度、特殊计数器”，命中时保底归零，边界就只需特殊计数器。
若命中后有其他信息继续影响未来，也必须纳入边界，不能为了缩小矩阵而丢失。

``HitTransitionBuilder(full_space, boundary_space, embed)`` 分别收集：

* ``add_miss(from_id, to_id, probability)``：未命中，终点属于完整空间。
* ``add_hit(from_id, boundary_id, probability)``：命中，终点属于边界空间。
* ``embed``：将边界状态映射到下一轮完整起点。可传每个边界对应的完整编号，
  或形状为 ``(完整状态数, 边界状态数)`` 的概率矩阵。

设未命中矩阵为 M、命中矩阵为 H、嵌入矩阵为 E，则每抽普通转移为
``P = M + E @ H``，周期核第 ``t >= 1`` 层为 ``H @ M**(t-1) @ E``。
构造时检查矩阵形状、有限性和非负性。``build(check=True)`` 或
``process.validate()`` 进一步检查每个完整起点的未命中与命中概率之和为 1，
以及 E 每列之和为 1。所有检查只报错，不修改概率。

``miss`` 和 ``hit`` 都使用 ``ProbabilityMatrixBuilder``，支持相同的边收集接口。
过程保存矩阵和状态空间的副本，构建后固定；对外访问矩阵、层矩阵或
``transition`` 获得独立副本，修改它们不会改变过程或内部缓存。
``HitProcessAnalysis`` 固定过程及计算上限，缓存结果对外返回副本。
更改规则或计算上限时构造新对象；改变初态只需向计算方法传入不同向量。

首次与连续命中
--------------------------------------

``first_hit(initial, max_steps)`` 接收完整空间初始概率向量，返回首次命中的
``StateDist``；花费 0 的质量为 0。``cycle_kernel(max_steps)`` 返回命中到命中的核。
``HitProcessAnalysis(process, max_steps)`` 缓存周期核及稳态结果，不绑定初态。
``first_state_dist(initial, initial_layer=None, return_remaining=False)`` 计算首件；
``nth_state_dist(count, initial, initial_layer=None, method="auto")`` 计算第 count 件累计花费。
``iter_state_dists`` 参数相同，依次产生第 1 至 count 件的联合分布，适合多件列表输出。
count 必须为正整数，零目标由调用者自行处理；任意初态结果不自动进入缓存。

.. code-block:: python

   import numpy as np
   from GGanalysis.markov import StateSpace, HitTransitionBuilder, HitProcessAnalysis

   # 首抽一半成功；失败后下一抽必成。每次成功后保底归零。
   full = StateSpace.from_shape([2])
   boundary = StateSpace.from_shape([1])
   builder = HitTransitionBuilder(full, boundary, embed=[0])
   builder.add_hit(0, 0, 0.5)
   builder.add_miss(0, 1, 0.5)
   builder.add_hit(1, 0, 1.0)
   process = builder.build()
   analysis = HitProcessAnalysis(process, max_steps=2)
   initial = full.delta([0])
   analysis.kernel.validate()
   assert np.allclose(analysis.first_state_dist(initial).marginal_cost().dist, [0, 0.5, 0.5])
   assert np.allclose(analysis.nth_state_dist(2, initial).marginal_cost().dist, [0, 0, 0.25, 0.5, 0.25])
   assert np.allclose(analysis.first_state_dist(full.delta([1])).marginal_cost().dist, [0, 1])
   assert abs(analysis.long_run_hit_rate - 2 / 3) < 1e-10

旧接口的迁移：移除构造函数中的 ``initial``、``initial_layer`` 参数，改在
``first_state_dist`` / ``nth_state_dist`` 调用时传入；``first_state_dist`` 由属性改为方法。
多个初态使用同一个分析对象即可，不需要为稳态计算提供占位初态。

``layered=True`` 是有条件的优化：完整状态必须按“连续未命中次数、边界状态”排列，
未命中只能前进一层，命中后由 E 回到第 0 层。不能仅因模型有保底就开启。
在此模式下指定 ``initial_layer`` 时，``initial`` 是该层上的边界长度向量，
不再是完整空间向量。首次接入时优先使用普通模式验证。
``validate()`` 检查分层结构，首次生成内部层矩阵时也执行检查；
不支持的非零边会报错，不会被优化路径静默丢弃。

从局部层矩阵自动构建
--------------------------------------

若机制天然按保底层描述，可用 ``HitProcess.from_layers`` 替代逐条填写完整编号。
传入每层的 ``B×B`` 命中和未命中矩阵，公共工具自动加上层偏移、组装完整稀疏矩阵、
启用分层计算并验证守恒。重复规则的层可引用同一矩阵；最后一层未命中必须为零。
完整空间、边界空间和 ``embed`` 仍由模型显式定义，工具不推断哪些信息需要保留。

.. code-block:: python

   from GGanalysis.markov import HitProcess

   layered = HitProcess.from_layers(
       full, boundary,
       hit_layers=[[[0.5]], [[1.0]]],
       miss_layers=[[[0.5]], [[0.0]]],
       embed=[0],
   )
   assert np.allclose(layered.cycle_kernel(2).coeff[:, 0, 0], [0, 0.5, 0.5])

例如异环只需构造一次位置移动矩阵，再按落点命中率逐行拆分为命中和未命中矩阵。
普通阶段、变格阶段分别复用各自的层矩阵，硬保底层使用原地命中的单位矩阵；
完整状态组装和周期核递推均交给公共工具。见 :doc:`../games/neverness_to_everness/index`。

有限步剩余概率
--------------------------------------

``first_hit(..., return_remaining=True)`` 返回 ``(StateDist, remaining)``；
``cycle_kernel(..., return_remaining=True)`` 返回 ``(StateKernel, remaining_by_state)``，
后者长度为边界状态数，分别对应各输入边界状态。默认返回类型不变。
剩余质量来自计算末端的存活向量，同时检查“已命中 + 剩余 = 初始质量”。
零步计算返回零命中质量和全部剩余质量。初始向量要求归一化。

.. code-block:: python

   joint, remaining = process.first_hit(full.delta([0]), 1, return_remaining=True)
   dist = joint.marginal_cost(tail_mass=remaining)
   dist.validate()
   assert dist.p_sum == 0.5
   assert dist.tail_mass == 0.5

``tail_mass`` 复用 ``FiniteDist`` 的已有字段，不增加虚构的花费位置。
剩余质量可能包含永不命中的概率，并不保证全部发生在有限的未来花费。
它不能补全尾部期望、方差或最终边界状态，也不会自动传播到后续状态核组合。
因此 ``FiniteDist.validate()`` 通过不意味着截断期望可以用于长期命中率。

开发验证使用含两个边界状态、三层保底、随机重置及跨边界未命中转移的小模型，
对比普通与分层路径在零步、部分截断、完整周期和不同初始层上的首次分布、周期核
与剩余质量。另用两抽保底模型验证解析分布和长期命中率，验证脚本放在本地 ``test/``。

稳态统计与截断
--------------------------------------

``post_hit_stationary`` 是命中时观察到的边界状态分布；
``steady_full_state`` 是随机观察一抽之前的完整状态分布，长周期会占用更多观察时间，
所以这两个量不可互换。``steady_dist`` 是稳态命中周期的花费分布，
``long_run_hit_rate`` 为其期望的倒数。

``max_steps`` 限制每次命中周期的计算范围，不是多个目标的累计预算。
有限保底时应覆盖所有起点的完整周期；无限尾部时需要检查每个起点的缺失质量，
增大上限验证结果稳定性。稳态计算会验证核的列质量，明显截断时会失败。
即使遗漏质量很小，也需检查长尾对期望的影响。可约边界链还需先明确所选闭合类，
不能直接假定唯一稳态。

API
--------------------------------------

``nth_state_dist`` 默认 ``strategy='step'`` 顺序应用周期核，显式指定
``'power'`` 才使用稠密状态核快速幂，不再按 12 件阈值切换。
``iter_state_dists`` 始终顺序产生每件结果，``method`` 只选择卷积后端。

``item_num_dist(pull, initial, initial_layer, multi_dist)`` 复用首件和周期核，
返回预算内命中数量分布；multi_dist 遍历投入 0 至 pull 步。
``event_count_state_dist(initial, steps)`` 从完整空间初态逐步统计数量及结束状态，
事件矩阵分别为 miss 和 embed @ hit，不受单周期 max_steps 上限影响。
前者依赖周期的预算内系数完整；后者保留完整状态，默认稀疏顺序传播。

.. autoclass:: GGanalysis.markov.hit_process.HitTransitionBuilder
   :members:
   :undoc-members:

.. autoclass:: GGanalysis.markov.hit_process.HitProcess
   :members:

.. autoclass:: GGanalysis.markov.hit_process.HitProcessAnalysis
   :members:
   :undoc-members:
