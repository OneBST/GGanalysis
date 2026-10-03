开发指南
======================================

本页面向项目开发者和代码 agent，说明如何复用工具、选择模型和新增游戏。
仓库级工作约定见根目录 ``AGENTS.md``；以下源码路径均相对于仓库根目录。

核心工具与定义位置
--------------------------------------

先选择现成模型，再考虑组合抽卡层；只有现有能力不能表达机制时才新增工具。
通用数学能力放公共模块，游戏特有规则放 ``GGanalysis/games/<game>/``。

.. list-table:: 工具入口
   :header-rows: 1
   :widths: 30 35 35

   * - 文件或目录
     - 主要定义
     - 用途
   * - ``GGanalysis/basic_models.py``
     - ``GachaModel``、``CommonGachaModel``；``BernoulliGachaModel``、``PityModel``、``DualPityModel``、``PityBernoulliModel`` 等
     - 模型基类及常见保底、概率筛选组合；集齐模型也在此处
   * - ``GGanalysis/gacha_layers.py``
     - ``GachaLayer``；``PityLayer``、``BernoulliLayer``、``MarkovLayer``、``DynamicProgrammingLayer``、``CouponCollectorLayer``
     - 将抽卡机制组合成模型
   * - ``GGanalysis/distribution_1d.py``
     - ``FiniteDist``、``linear_p_increase``、``p2dist``、``independent_item_num_dist``
     - 一维分布、保底概率表转换、独立花费卷积和固定抽数下的数量分布
   * - ``GGanalysis/state_distribution.py``
     - ``StateDist``、``StateKernel``
     - 花费与状态的联合分布，以及保留状态依赖的阶段组合
   * - ``GGanalysis/markov/``
     - ``StateSpace``、``TransitionBuilder``、``HitTransitionBuilder``、``HitProcessAnalysis``；``StateRewards``、``group_mass``、``PriorityPityChain``
     - 状态编码、转移构建、命中过程、稳态与事件奖励统计、多稀有度保底优先级
   * - ``GGanalysis/scored_item/``
     - ``scored_item.py``、``scored_item_tools.py``
     - 装备评分分布及组合、筛选
   * - ``GGanalysis/simulation/``
     - ``Statistics``、``HoyoItemSim``、``HoyoItemSetSim``
     - 可复用的统计工具及装备抽样模拟
   * - ``GGanalysis/reverse_engineering/``
     - ``LinearAutoCracker``、词条权重似然函数
     - 抽卡与装备机制的逆向工程；不限于参数推断
   * - ``GGanalysis/gacha_plot.py``、``GGanalysis/plot_tools.py``
     - ``DrawDistribution`` 等
     - 复用分布展示与绘图样式

模型选型见 :doc:`reference_manual/basic_models`，抽卡层约定见 :doc:`reference_manual/gacha_layers`。
状态类型见 :doc:`reference_manual/state_distribution`，转移及命中过程见
:doc:`reference_manual/markov`、:doc:`reference_manual/hit_process`。
组合示例见 :doc:`start_using/custom_gacha_model`，非 IID 教程见
:doc:`start_using/stateful_models`，一维 API 见 :doc:`reference_manual/basic_tools`。
表中未收录到参考手册的接口，先阅读对应源码的 docstring；不要根据名字猜测语义。

公开入口与包命名
--------------------------------------

模型、抽卡层和 ``markov`` 的公开工具均可从根包 ``GGanalysis`` 导入；
也可以从对应模块或领域包导入同一对象。根包采用按需导出。
``GeneralCouponCollection``、``HitProcess``、``StateRewards`` 等同时由
``GGanalysis`` 和 ``GGanalysis.markov`` 提供，领域入口不要求用户了解实现文件。

装备评分、模拟和逆向工程的目录分别为 ``scored_item``、``simulation``、
``reverse_engineering``，使用小写名称；旧的大小写目录名不再提供。
顶层包名保留 ``GGanalysis``，模块名 ``basic_models.py``、``gacha_layers.py``
和目录名 ``markov`` 保持不变。

模型选择与独立性假设
--------------------------------------

建模前明确目标事件、花费单位、初始条件，以及获得目标后哪些状态重置、哪些继承。
这里的 IID 指“各次获得目标所需的花费独立同分布”，不是“每一抽独立同分布”。
带保底的每抽概率可以不同，但若每次获得目标后过程完整重置，各轮花费仍可为 IID。

* **首次受垫抽影响，后续周期 IID**：优先使用 ``basic_models.py`` 的现成模型。
  ``CommonGachaModel`` 按首次条件分布乘以后续完整分布的卷积幂，组合多个目标。
* **各阶段独立但不同分布**：分别计算各阶段的 ``FiniteDist``，再逐个卷积；
  不使用同一个分布的幂代替不同阶段。
* **获得目标后仍保留影响未来的状态**：使用 ``StateDist`` 与 ``StateKernel``
  衔接阶段。可通过 ``markov/hit_process.py`` 构建命中过程和周期核。
  不能先丢掉结束状态，再对单次边缘花费分布做卷积幂。
* **固定抽数下获得多少个道具**：IID 周期且每抽最多获得一个目标时，可用
  ``independent_item_num_dist``，首次条件分布通过 ``c_dist`` 传入。
  不满足条件时使用相应状态递推。
* **集齐问题**：先查 ``basic_models.py`` 的集齐模型和
  ``markov/coupon_collection.py``，区分等概率与不等概率物品。
* **长期概率或多稀有度相互挤占**：查阅 ``markov/analysis.py`` 和
  ``markov/priority_pity.py``，明确初始分布与稳态解的适用条件。

``GachaModel`` 目前是空基类，不会自动检查接口或实现计算。
``GachaLayer`` 返回 ``(完整分布, 条件分布)``，自定义层应保持该约定。
使用 ``MarkovLayer`` 只表示某一层由马尔可夫过程计算；若外层仍使用
``CommonGachaModel`` 的多目标组合，仍须满足后续周期 IID 假设。

统一采用已有数据语义：``FiniteDist.dist[t]`` 表示花费恰为 ``t`` 的概率，
保底表 ``pity_p[k]`` 表示前面未获得目标时第 ``k`` 抽的条件概率（索引 0 为占位）。
``StateDist.coeff[t, s]`` 表示花费与状态的联合概率；
``StateKernel.coeff[t, end, start]`` 表示从起始状态到结束状态的联合条件概率。
状态转移采用列向量约定，``kernel_b @ kernel_a`` 表示先执行 A，再执行 B。

新增游戏的最低要求
--------------------------------------

最低结构如下；不要为没有需求的功能创建空文件：

.. code-block:: text

   GGanalysis/games/<game>/
       __init__.py          # 对外导出
       gacha_model.py       # 可调用模型及必要配置
   docs/source/games/<game>/
       index.rst            # 最小使用文档

新增游戏至少完成以下内容：

1. **模型入口**：在 ``gacha_model.py`` 提供可调用模型，优先实例化已有模型。
   语义适用时沿用 ``item_num``、``multi_dist``、``item_pity``、``up_pity``。
   常规花费查询返回 ``FiniteDist``；``multi_dist=True`` 的列表索引 ``k``
   表示获得 ``k`` 个目标的累计花费分布，索引 0 为零花费分布。
   零目标应返回零花费分布；若需不同返回结构，明确记录，而不是暗中改变语义。
2. **公开导出**：明确 ``__all__``，通过游戏包 ``__init__.py`` 暴露公开接口。
   实现辅助函数不作为默认公开接口。
3. **机制假设**：注明规则来源或推断依据、适用版本、目标事件、初始状态、
   重置与继承规则，以及近似和未覆盖机制。概率表与规则常量单处定义，
   模型、绘图和模拟共同引用；配置复杂时再拆为 ``gacha_data.py``。
4. **最小文档**：说明支持范围、机制假设、参数含义、返回含义和最小调用示例；
   在 ``docs/source/games/index.rst`` 的 ``toctree`` 中加入新游戏。

简单保底及层组合参考 ``GGanalysis/games/honkai_star_rail/gacha_model.py``；
跨次状态依赖及配置、状态核拆分参考 ``GGanalysis/games/neverness_to_everness/``。
参考组件组织与接口语义，不照搬其他游戏的参数或机制假设。
稳态分析、绘图、装备模型和模拟器按需添加，不是接入的最低要求。

正式说明使用中文 RST。新增公开接口的 docstring 写清参数、返回值和假设，
参数说明统一采用 NumPy 风格；API 页面按需使用 autodoc 引用，避免手工复制签名。
机制与接口变更同步更新对应文档，不因格式统一批量重写无关旧代码。

验证与本地测试
--------------------------------------

测试、基准脚本和临时验证报告统一放在已被 Git 忽略的 ``test/``，暂不提交。
可复用的模拟功能属于正式源码，例如 ``GGanalysis/simulation/`` 或游戏内的模拟器。
重要数学假设、验证结论和适用限制保留在正式文档中，不将临时运行日志搬入文档。

常规参数变化与已有模型组合，优先检查概率质量、零目标、保底边界和已知结果；
新算法或状态转移可用小规模枚举、退化到已知模型、不同确定性算法对照验证。
记录截断尾部质量或误差，不用强制归一化掩盖转移遗漏。

只有需要独立交叉验证，或状态规模使确定性计算不可行时，才使用模拟。
模拟应记录随机种子、样本量、对比指标及置信区间或标准误差；
验证逻辑尽量独立于被测算法，同时共享规则常量，避免维护两份游戏定义。
模拟吻合只能支持实现一致性，不能证明游戏机制假设正确。

修改文档后，从仓库根目录构建并检查新增页面、侧边栏入口及链接：

.. code-block:: console

   python -m sphinx -M html docs/source ../ggdoc

构建依赖见 ``docs/requirements.txt``；Windows 也可在 ``docs/`` 中使用
``.\make.bat html``。两者均将网页输出到同级目录 ``ggdoc/html/``，
中间文件输出到 ``ggdoc/doctrees/``；不要将构建产物写入源码目录。
报告实际检查结果，区分新增问题和既有警告。
