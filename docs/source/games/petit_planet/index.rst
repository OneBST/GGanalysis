星布谷地（三测）
================

本模型按连接测试各池概率表、奇兑好礼价格表及跨季继承说明计算。每抽消耗记为 1，
忽略露芽返利。同类上新池的保底和兑换币跨季保留。长期计算假设每季均为新款，
沿用同一套概率与价格。

.. code-block:: python

   from GGanalysis.games.petit_planet import single_season, guaranteed_pulls, long_run

   dist = single_season("curio_furniture")
   print(dist[100], dist.cdf[100], dist.exp)  # 恰好第 100 抽、100 抽以内、平均抽数
   print(guaranteed_pulls("curio_furniture"))  # 220
   pure = single_season("curio_clothing", "never")
   print(pure.exp, pure.tail_mass)  # 完整纯抽期望、720 抽后仍缺件的概率
   one = single_season("curio_clothing", target={("rainbow", 1)}, balance=80)
   steady = long_run("curio_clothing", max_pulls=720)
   print(steady.mean_pulls, steady.stop_dist.tail_mass)

收集口径
--------

.. list-table::
   :header-rows: 1

   * - 卡池键
     - 单期最优兑换目标（各一件）
     - 空仓保证抽数
   * - curio_clothing
     - 18 件服饰，魔发泡泡作为挑染耗材单列
     - 720
   * - curio_furniture
     - 23 件家具及围栏创想卡；视为有卡即可制作围栏
     - 220
   * - seasonal_clothing
     - 8 件服饰，共 745 小星线
     - 116
   * - standard_furniture
     - 18 件成品，共 852 小积木
     - 132

上新池每十抽至少一次彩金，提前命中立即重置，保底时按对应物品的基础概率权重抽取。
普通池以 20、22、24 档为保底目标，100 档不重置计数。
普通池只产材料，禁止兑换时无法直接获得成品。上新家具的纯抽参考曲线默认收
24 件可抽成品；若明确要求没有持有的配方卡且禁止兑换，则目标不可达。

指定单件使用 (组键, 从 1 开始的编号)。服饰组为
``rainbow, gold_170, gold_150, gold_60, purple, blue``；家具组为
``rainbow, gold, purple``。家具第 12 件紫档代表围栏，其余同组成员可以任意固定编号。
例如两件彩装任选其中一件作为预先指定目标时，都可以用编号 1 表示；
同时指定两件则传入两个编号。单期允许传入已拥有物品、保底、余额和已有卡状态。
普池可以直接指定所需材料总额。

分布与策略
----------

单期允许兑换时，首次够换齐即完成。推迟兑换能保留自然出货的机会，当前各物品的
首件价格均覆盖重复返币，所以该策略逐路径最早完成。纯抽及材料池用首次命中转移；
纯抽期望单独解未截断的线性方程。上新池允许兑换时，以已出彩金次数为条件，
两档内部的物品序列独立。分别按各档拥有数和累计有效收益推进，汇总下一抽跨过
剩余兑换门槛的概率。首次获得目标按兑换价计收益，非目标首件记零，重复品按返币计。
浮点数可能将极小的尾部单点概率下溢为零，保证抽数另由最坏合法路径确定。

默认截止为表中各池的保证抽数，纯抽也采用同一截止。返回的
``FiniteDist.tail_mass`` 为截止后仍未完成的质量；零抽已完成和永久不可达分别
落在零抽和尾部。缩短截止后，兑换分布用 ``tail_mass`` 查看遗漏质量，完整均值
请用覆盖保证抽数的截止计算。

无限跨季纯抽保留每季结束时的保底。兑换策略最小化平均每季抽数，并要求平均
有效收益覆盖每季全部目标的兑换成本。算法按季初和季末保底平衡各停手策略的
使用次数，并用最优停抽子问题生成候选策略。策略表每个拥有数组合与保底对应
一个继续概率，组成员及顺序在 ``group_items`` 中。资源松弛模型允许用启动储备
渐近逼近最优值；真实余额不足时应继续筹币。``stop_dist`` 是所求平稳策略的
季内停手分布，可以在第零抽停手。若最优值需要无限期积攒重复返币的极限策略，
该分布留空。普池的长期模型按同样清单反复开放、余额和保底继承计算。

验证基准
--------

从零开始，服饰与家具允许兑换的单期期望分别为 171.416943 和 108.199819。
家具分块递推与独立的全状态稀疏转移计算到 220 抽，最大单点绝对差小于
:math:`10^{-14}`。服饰纯抽完整期望为 2212.817279，720 抽后尾概率为 0.851827285。
无限期全收集的最优平均抽数为服饰 158.112370、家具 104.744160。

接口
----

.. autofunction:: GGanalysis.games.petit_planet.single_season

.. autofunction:: GGanalysis.games.petit_planet.guaranteed_pulls

.. autofunction:: GGanalysis.games.petit_planet.long_run

.. autoclass:: GGanalysis.games.petit_planet.LongRunResult
