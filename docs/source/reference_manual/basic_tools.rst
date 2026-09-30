一维分布与基础工具
======================================

定义位置：``GGanalysis/distribution_1d.py``。``FiniteDist.dist[t]`` 表示非负整数
花费恰为 t 的概率质量，不是第 t 抽的条件成功率。保底概率表用 ``p2dist``
或 ``FiniteDist.from_pity_p`` 转成花费分布。

按需导入
--------------------------------------

``from GGanalysis import FiniteDist`` 与
``from GGanalysis.distribution_1d import FiniteDist`` 均只加载一维分布所需的
NumPy 等基础依赖，不会连带初始化抽卡层、状态分布和马尔可夫工具。
构造分布、求概率和、期望、方差、CDF 及保底表转换不需要加载 SciPy。

首次卷积或 FFT 运算才加载对应 SciPy 函数，后续复用已加载的函数。
卷积仍使用原始 ``scipy.signal.convolve`` 的自动算法选择，高精度接口
仍指定直接卷积；按需导入不改变计算定义或精度策略。首次卷积会承担
延后的依赖加载时间，因此导入提速不等于首次完整卷积任务同等幅度提速。
星号导入会请求全部历史导出名称，包括 SciPy 函数，仍会加载相关依赖；
只需要一维分布时推荐显式导入 ``FiniteDist``。

构造与运算
--------------------------------------

``FiniteDist`` 接受列表、数组或另一个分布。``delta(t)`` 表示确定花费，
``from_cdf`` 从累计概率构造，``mixture`` 按权重混合多个分布。
构造器不自动归一化，概率用途应调用 ``validate()`` 检查。

.. code-block:: python

   import numpy as np
   from GGanalysis.distribution_1d import FiniteDist

   dist = FiniteDist.from_pity_p([0, 0.5, 1])
   dist.validate()
   assert np.allclose(dist.dist, [0, 0.5, 0.5])
   assert dist.exp == 1.5
   assert dist.var == 0.25
   assert np.allclose((dist ** 2).dist, [0, 0, 0.25, 0.5, 0.25])
   mixed = FiniteDist.mixture([FiniteDist.delta(1), FiniteDist.delta(3)], [0.25, 0.75])
   assert mixed.exp == 2.5

.. list-table:: 运算含义
   :header-rows: 1

   * - 表达式
     - 含义
   * - ``a * b``
     - 两个独立花费之和的分布；不要求同分布
   * - ``a ** n``
     - n 个 IID 花费之和；n 为非负整数，0 次为零花费分布
   * - ``a + b``
     - 概率系数逐项相加，不是随机变量相加
   * - ``w * a``
     - 概率质量乘权重，不是将花费乘 w
   * - ``a ** count_dist``
     - 按次数分布混合卷积幂；概率解释要求次数独立于各次 IID 花费

``mixture`` 不强制权重非负或和为 1，由调用者保证概率解释。
CDF 的第 t 项为花费不超过 t 的概率。``quantile_point(q)`` 搜索 CDF 首次达到 q
的位置；若截断数组未达到 q，返回的是数组长度，不能视为已求得的分位点。

只读视图与修改
--------------------------------------

``dist.dist`` 与 ``dist.cdf`` 为只读视图，避免绕过缓存失效机制。
通过对象下标赋值或 ``set_dist`` 修改分布；修改会使旧的统计量、卷积幂缓存、
尾质量及附加信息失效。``set_dist`` 可同时提供与新数组匹配的统计信息。

.. code-block:: python

   from GGanalysis.distribution_1d import FiniteDist

   dist = FiniteDist([0, 0.5, 0.5])
   old_mean = dist.exp
   dist[1:3] = [0.25, 0.75]
   assert dist.exp == 1.75
   dist.set_dist([0, 1], tail_mass=0)
   dist.validate()
   assert dist.exp == 1

截断、统计量与条件化
--------------------------------------

``p_sum`` 是数组内的质量；``tail_mass`` 是已知但未存入数组的质量，未知时为
``None``，不是 0。``validate`` 在尾质量已知时检查两者之和，否则只检查数组质量。
``metadata`` 可记录算法及来源，但不应假设所有运算都保留这些附加信息。

截断尾质量不足以恢复期望和方差。部分模型会显式保存完整分布的理论统计量，
此时 ``exp`` / ``var`` 可以不等于截断数组直接计算的结果；若无显式统计量且
数组质量偏离 1 超过内部容差，自动计算的期望或方差可能为 ``nan``。

``normalized()`` 只对数组内质量归一化，并将新对象尾质量设为 0。
它表示在保留事件上条件化，不是恢复截断尾部。零质量无法归一化。
``from_cdf`` 长度为 1 时保留历史行为，返回零花费必然分布；若需表示仅在
零花费保留部分质量，应直接构造 ``FiniteDist([mass], tail_mass=...)``。

.. code-block:: python

   from GGanalysis.distribution_1d import FiniteDist

   partial = FiniteDist([0, 0.5, 0.25], tail_mass=0.25)
   partial.validate()
   assert partial.p_sum == 0.75
   conditional = partial.normalized()
   assert abs(conditional.exp - 4 / 3) < 1e-12

数量分布与集齐
--------------------------------------

``independent_item_num_dist(f_dist, pull, c_dist=None)`` 将等待花费转成固定抽数内
的目标数量分布，要求后续周期 IID、每抽至多一个目标；首次条件分布可单独指定。
``calc_item_num_dist`` 从累计获得第 k 个目标的花费分布列表转换数量分布；
传入列表应覆盖可能的目标数，否则无法完整区分较大的数量。
带状态依赖的数量计算应保留状态，不强行使用 IID 接口。

首次和后续等待花费都必须是正整数，零花费概率必须为 0。
数量转换要求输入保留查询抽数以内的全部概率系数；数组外的质量视为等待
尚未结束，不因数组较短就把累计概率补成 1。若截断发生在查询范围以内，
仅凭 ``tail_mass`` 无法恢复准确的数量分布。``calc_item_num_dist`` 的最后
一项表示至少获得 k 个，其余项表示恰好获得对应数量。

.. code-block:: python

   from GGanalysis.distribution_1d import independent_item_num_dist

   # 每五抽必得一个目标，三抽内必定尚未获得。
   count = independent_item_num_dist(FiniteDist.delta(5), 3)
   assert count.dist.tolist() == [1.0, 0.0, 0.0, 0.0]

不等概率集齐的低层工具定义于 ``GGanalysis/markov/coupon_collection.py``；
常规用户优先使用 :doc:`basic_models` 中的集齐模型。

数值计算与边界
--------------------------------------

方差采用中心化公式，避免大均值、小方差时由二阶矩相减造成的精度损失。
``calc_variance`` 按归一化概率分布解释输入；对于缺失尾部的系数，不能据此
恢复完整分布的方差。统计属性仍保留模型显式提供的理论期望与方差。

熵计算约定零概率项贡献为零，包括分布内部的零系数。期望非正时
``entropy_rate`` 返回 ``nan``；``randomness_rate`` 仅在期望大于 1 时定义，
因为期望为 1 时参考伯努利熵为零，期望小于 1 时参考概率超出有效范围。

``accurate_pow(n)`` 使用直接卷积，要求 n 为非负整数；0 次幂返回零花费
必然分布。普通整数幂保留快速幂与缓存策略，普通卷积允许自动选择 FFT；
FFT 的极小尾概率有绝对误差底噪。需要计算很小的尾概率时使用
``accurate_pow`` 或 ``accurate_conv``，两者仍受浮点下溢限制。

保底表转换通过生存概率的累积乘积计算，保留末端未命中质量；统计量求和
采用数组运算，广义卷积幂预分配结果数组。实现不强制归一化来掩盖质量误差。
确定性验证覆盖了独立等待时间递推、伯努利数量分布、直接卷积、截断与零次幂、
零概率熵以及大均值小方差边界。

API
--------------------------------------

.. autoclass:: GGanalysis.distribution_1d.FiniteDist
   :members:

.. automodule:: GGanalysis.distribution_1d
   :members: linear_p_increase, pad_zero, cut_dist, calc_expectation, calc_variance, dist2cdf, cdf2dist, p2dist, dist2p, p2exp, p2var, calc_item_num_dist, independent_item_num_dist, calc_bernoulli_obtain, accurate_conv

.. autoclass:: GGanalysis.markov.coupon_collection.GeneralCouponCollection
   :members:

.. autofunction:: GGanalysis.markov.coupon_collection.get_equal_coupon_collection_exp
