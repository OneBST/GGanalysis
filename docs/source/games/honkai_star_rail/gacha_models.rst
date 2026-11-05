崩坏：星穹铁道抽卡模型
========================

GGanalysis 使用基本的抽卡模板模型结合 `基于1500万抽数据统计的崩坏：星穹铁道抽卡系统 <https://www.bilibili.com/video/BV1nt421w7FE/>`_ 定义了一系列可以直接取用的抽卡模型。

.. attention:: 

   崩坏：星穹铁道的四星保底不会被五星重置，但与五星耦合时仍会在综合概率上产生细微的影响。此处的模型没有考虑四星和五星的耦合。

   崩坏：星穹铁道的限定角色卡池与限定光锥卡池 **实际UP概率显著高于官方公示值** ，模型按照统计推理得到值建立。

   崩坏：星穹铁道常驻跃迁中具有“平稳机制”，即角色和光锥两种类型的保底，GGanalysis 包没有提供这类模型，有需要可以借用 `GGanalysislib 包 <https://github.com/OneBST/GGanalysislib>`_ 为原神定义的模型进行计算。角色活动跃迁及光锥活动跃迁中的四星道具也有此类机制，由于其在“UP机制”后生效，对于四星UP道具抽取可忽略。

参数意义
------------------------

    - ``item_num`` 需求物品个数，由于 sphinx autodoc 的 `bug <https://github.com/sphinx-doc/sphinx/issues/9342>`_ 在下面没有显示

    - ``multi_dist`` 是否以列表返回获取 1-item_num 个物品的所有分布列

    - ``item_pity`` 道具保底状态，通俗的叫法为水位、垫抽

    - ``up_pity`` UP道具保底状态，设为 1 即为玩家所说的大保底

卷积精度与计算规模
------------------------

模型通过首次条件分布与后续 IID 花费的卷积幂计算多个目标。默认快速幂
内部的卷积由 SciPy 自动选择直接算法或 FFT，因此默认算法也不保证极小
尾概率始终具有较小的相对误差。直接在频域计算幂可减少变换次数，但
逆变换中的浮点相消会在极小概率处产生正负底噪；将负值裁为零不能恢复
真实尾概率，也不能用强制归一化消除该误差。

针对本仓库模型的确定性对照覆盖了 1、2、7 个 UP 五星角色，1、5 个 UP
五星光锥，70 垫抽且大保底时获取 7 个角色，7 角色加 5 光锥，以及 100
角色压力场景。以非负项直接卷积为参考，单次 FFT 幂候选在这些场景下的
累计概率最大绝对误差低于 ``1e-12``，50%、90%、95%、99%、99.9% 和
99.999% 分位点均一致。统计量比较使用概率系数重新计算，未使用模型已知
的理论期望或方差替代数值检验。此验证针对当前代码参数，不是对游戏规则
的新验证，也不构成任意参数下的误差上界。

极小尾概率需要区别处理。例如从零垫抽、无大保底开始，七个 UP 五星角色
的总花费至少为 1187 抽的概率，70 位十进制计算约为 ``5.38549686e-31``；
测试中的单次 FFT 幂候选得到约 ``6.84e-18``，而当前默认计算在该规模
使用直接卷积并保留了此尾概率。右尾应由概率系数反向累加，避免用
``1 - cdf`` 产生另一层相消误差。

缓存只有复用同一个 ``FiniteDist`` 对象时才能跨次命中；重复调用模型
会重建中间分布，不能把底层幂缓存命中的耗时作为整个模型的重复调用耗时。
``multi_dist=True`` 使用逐次卷积，单独替换整数幂实现不会加速这条路径。

基本模型
------------------------

**角色活动跃迁及常驻跃迁获得五星道具的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.common_5star

**角色活动跃迁及常驻跃迁获得四星道具的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.common_4star

角色活动跃迁模型
------------------------

**角色活动跃迁获得UP五星角色的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_5star_character

**角色活动跃迁获得任意UP四星角色的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_4star_character

**角色活动跃迁获得特定UP四星角色的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_4star_specific_character

.. code:: python

    import GGanalysis.games.honkai_star_rail as SR
    # 崩坏：星穹铁道角色池的计算
    print('角色池在垫了20抽，有大保底的情况下抽3个UP五星抽数的分布')
    dist_c = SR.up_5star_character(item_num=3, item_pity=20, up_pity=1)
    print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)

光锥活动跃迁模型
------------------------

**光锥活动跃迁获得五星光锥的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.common_5star_weapon

**光锥活动跃迁获得UP五星光锥的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_5star_weapon

**光锥活动跃迁获得四星光锥的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.common_4star_weapon

**光锥活动跃迁获得UP四星光锥的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_4star_weapon

**光锥活动跃迁获得特定UP四星光锥的模型**

.. automethod:: GGanalysis.games.honkai_star_rail.gacha_model.up_4star_specific_weapon

幻乐秀场时装抽取
------------------------

``Phantasia Showstage`` （幻乐秀场）模型计算从未参与活动开始，首次获得
``Star Lotto`` （群星乐透）时装的花费。英文名称见
`HoYoverse 4.7 新闻稿 <https://www.gamespress.com/Honkai-Star-Rail-Version-47-Now-Lay-Laughter-to-All-St-wwwhoyoversecom>`_，
概率与消耗依据 2026-10-30 提供的官方活动说明、游戏内公示截图。
活动时间为 2026-11-11 的 4.7 更新后至 2026-12-23 03:59。

.. list-table:: 时装逐抽条件概率与消耗
   :header-rows: 1

   * - 抽取次数
     - 尚未获得时装时的命中概率
     - 本次邀请函消耗
     - 累计星琼消耗
   * - 1
     - 0.05%
     - 0
     - 0
   * - 2
     - 0.25%
     - 3
     - 300
   * - 3
     - 1%
     - 5
     - 800
   * - 4
     - 1.5%
     - 8
     - 1600
   * - 5
     - 2%
     - 12
     - 2800
   * - 6
     - 5%
     - 15
     - 4300
   * - 7
     - 25%
     - 20
     - 6300
   * - 8
     - 100%
     - 24
     - 8700

每抽获得一个奖项并将其移除；抽到时装时同时获得所有剩余奖品，活动结束。
模型只计算首次获得时装，不支持重复获取或跨期继承，不需要假设非时装奖品
的后续概率如何分配。邀请函均按每张 100 星琼折算，首次免费；不扣除
邀请函消耗计入「万象和鸣」所带来的其他奖励价值。

设第 k 抽的条件命中概率为 p[k]，则恰好在该抽获得时装的概率为
``p[k] * prod(1 - p[j], j < k)``。将该概率映射到累计邀请函或星琼消耗，
即可得到花费分布；邀请函与星琼分布的零花费概率均为 0.0005。
这是有限精确递推（仅有浮点舍入误差），不使用模拟、截断或强制归一化。

期望消耗为 **7623.36 星琼，等效 47.65 张星轨专票（160 星琼/张）**。
该折算票数表示跃迁资源价值，与本活动的实际抽取次数不同。
6300 星琼内获得时装的概率约为 32.11433568%，以 8700 星琼保底获得的
概率约为 67.88566432%。50%、90%、99% 分位点均为 8700 星琼。
本地有理数生存递推已交叉验证分布及期望，并检查概率质量、首抽免费、
第八抽必得、花费映射和预算区间内累计概率不变。

.. autofunction:: GGanalysis.games.honkai_star_rail.gacha_model.phantasia_showstage

.. code-block:: python

    from GGanalysis.games.honkai_star_rail import phantasia_showstage

    pull_dist = phantasia_showstage()  # 包含免费首抽
    jade_dist = phantasia_showstage('stellar_jade')
    print('期望抽数', pull_dist.exp)
    print('期望星琼', jade_dist.exp)
    print('6300 星琼内获得的概率', jade_dist.cdf[6300])

从仓库根目录运行绘图脚本：

.. code-block:: console

    python -m GGanalysis.games.honkai_star_rail.figure_plot_phantasia_showstage

脚本输出各消耗档位的概率与期望，在 ``figure/`` 中保存
``phantasia_showstage_star_lotto.png`` 和同名 SVG。
上图为八个实际星琼档位的等宽柱状图，每根柱子标注恰好在该档获得时装的
概率；下图为无面积填充的累计概率阶梯图，每个档位标注星琼和累计概率，
上下绘图区高度比为 5:4，左右两端各留出 500 星琼空间，
竖向次网格线按 100 星琼的固定间隔绘制。
绘图复用 ``DrawDistribution.show_two_graph`` 模板的布局、网格和文字样式，
调整数据图元、x 轴档位及期望标注，同时显示星琼与按 160 星琼折算的星轨专票数
（均保留两位小数）。
