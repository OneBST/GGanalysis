## 采用模型

工具包里原神模块基于的抽卡模型见[原神抽卡全机制总结](https://bbs.nga.cn/read.php?tid=26754637)，是非常准确的模型。为了实现方便，工具包中对UP四星角色、UP四星武器、UP五星武器时不考虑从常驻中歪到的情况，计算四星物品时忽略四星物品被五星物品顶到下一抽的情况。这些近似实际影响很小。

绘制概率图表见[原神抽卡概率工具表](https://bbs.nga.cn/read.php?tid=28026734)

`up_5star_character` 采用捕获明光双计数器猜想：`cr_counter` 为 A，新增
`cr_counter_b` 为 B；B 的触发概率是拟合估计。`small_pity_up_rate()` 返回
长期小保底 UP 胜率，默认参数约为 55.2691677%，不计大保底。

`standard_5star_character`、`standard_5star_weapon` 计算包含类别平稳机制的
常驻五星角色/武器抽数分布，可传 `character_pity`、`weapon_pity` 和
`item_pity`；这两个类别进度的较小值须等于五星水位。机制参数及用例见正式文档。

## 其他

写了一个估算总氪金数的小程序`GetCost.py`，比较粗糙而且没怎么检查，可以用着玩一下，以后会仔细想想放哪里。

`predict_next_type.py` 用于展示下一个常驻五星的类别概率，与正式类别模型共用权重函数。
