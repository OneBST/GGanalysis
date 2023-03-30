# 抽卡游戏概率分析工具包-GGanalysis

点此查看[在线文档](https://onebst.github.io/GGanalysis/)

本工具包为快速构建抽卡游戏抽卡模型设计，通过引入“抽卡层”，以抽卡层的组合快速实现复杂的抽卡逻辑，并以较低的时间复杂度计算抽卡模型对应的分布。除了计算分布，工具包还提供一些将计算结果可视化的绘图工具，并计划加入更多的针对各类抽卡问题的计算工具和设计工具。

近期加入了对有分值道具的相关功能，可以对类似原神圣遗物等问题进行建模。但请注意，这部分代码随时可能发生变动，接口可能大幅改动。

工具包名称从 `GGanalysisLite` 改为 `GGanalysis`，此前的 `GGanalysis` 包暂更名为 `GGanalysislib`。

## 安装方法

本工具包的依赖库很简单，只需要安装`numpy`和`scipy`即可。如果需要使用工具包提供的画图代码，还需安装 `matplotlib`。

``` shell
pip install numpy scipy matplotlib
```

工具包目前还在开发中，如想安装本工具包可以打开终端输入以下指令，安装完成后下载文件可以删除。

```shell
git clone https://github.com/OneBST/GGanalysis.git
cd GGanalysis
pip install .
```

画图时需要安装[思源黑体](https://github.com/adobe-fonts/source-han-sans)，安装[对应版本](https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip)后即可使用。

## 支持抽卡层

1. `Pity_layer` 保底抽卡层，实现每抽获取物品的概率仅和当前至多多少抽没有获取过物品相关的抽卡模型。
2. `Bernoulli_layer` 伯努利抽卡层，实现每次抽卡获取物品的概率都是相互独立并有同样概率的抽卡模型。
3. `Markov_layer` 马尔可夫抽卡层，实现每次抽卡都按一定概率在状态图上进行转移的抽卡模型。保底抽卡层是马尔科夫抽卡层的特例。
3. `Coupon_Collector_layer` 集齐道具层，实现每次抽卡随机获得某种代币，代币有若干不同种类，当集齐一定种类的代币后获得物品的抽卡模型。（注意：目前集齐道具层的功能已经可以使用，但还未经过充分的测试）

## 使用方法


**使用定义好的抽卡模型计算抽卡所需抽数分布**

``` python
# 计算抽卡所需抽数分布律 以原神为例
import GGanalysis.games.genshin_impact as GI
# 原神角色池的计算
print('角色池在垫了20抽，有大保底的情况下抽3个UP五星抽数的分布')
dist_c = GI.up_5star_character(item_num=3, pull_state=20, up_guarantee=1)
print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)

# 计算抽卡所需抽数分布律 以明日方舟为例
import GGanalysis.games.arknights as AK
# 普池双UP的计算 item_num是要抽多少个 pull_state是当前垫了多少抽，从零开始填0就行
dist_c = AK.dual_up_specific_6star(item_num=3, pull_state=20)
print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)
```

**自定义抽卡模型**

``` python
import GGanalysis as gg
# 定义一个星级带软保底，每个星级内有5种物品，想要获得其中特定一种的模型
# 定义软保底概率上升表，第1-3抽概率0.1，第4抽概率0.6，第5抽保底
pity_p = [0, 0.1, 0.1, 0.1, 0.6, 1]

# 采用预定义的保底伯努利抽卡类
gacha_model = gg.PityBernoulliModel(pity_p, 1/5)
# 根据定义的类计算从零开始获取一个道具的分布，由于可能永远获得不了道具，分布是截断的
dist = gacha_model(item_num=1, pull_state=0)

# 从头定义抽卡类
# 保底伯努利抽卡类
class MyModel(gg.CommonGachaModel):
    # 限制期望的误差比例为 1e-8，达不到精度时分布截断位置为 1e5
    def __init__(self, pity_p, p, e_error = 1e-8, max_dist_len=1e5) -> None:
        super().__init__()
        # 增加保底抽卡层
        self.layers.append(gg.PityLayer(pity_p))
        # 增加伯努利抽卡层
        self.layers.append(gg.BernoulliLayer(p, e_error, max_dist_len))
# 根据定义的类计算从零开始获取一个道具的分布，由于可能永远获得不了道具，分布是截断的
gacha_model = MyModel(pity_p, 1/5)
# 关于 pull_state 等条件输入，如有需要可以参考预置类中 __call__ 和 _build_parameter_list 的写法
dist = gacha_model(item_num=1)
```

**使用转移矩阵方法计算复合类型保底的概率**

``` python
# 以明日方舟为例，计算明日方舟六星、五星、四星、三星物品耦合后，各类物品的综合概率
import GGanalysis as gg
import GGanalysis.games.arknights as AK
# 按优先级将道具概率表组成列表
item_p_list = [AK.pity_6star, AK.pity_5star, [0, 0.5], [0, 0.4]]
AK_probe = gg.PriorityPitySystem(item_p_list, extra_state=1, remove_pity=True)
# 打印结果，转移矩阵的计算可能比较慢
print(AK_probe.get_stationary_p())
# [0.02890628 0.08948246 0.49993432 0.38167693]
```

**使用迭代方法计算平稳分布后n连抽获得k个道具概率**

``` python
# 以原神10连获得多个五星道具为例
import GGanalysis as gg
import GGanalysis.games.genshin_impact as GI
# 获得平稳后进行10连抽获得k个五星道具的分布
ans = gg.multi_item_rarity(GI.pity_5star, 10)
```

**绘制简略的概率质量函数图及累积质量函数图**

``` python
# 绘图前需要安装 matplotlib 以及需要的字体包
import GGanalysis.games.genshin_impact as GI
# 获得原神抽一个UP五星角色+定轨抽特定UP五星武器的分布
dist = GI.up_5star_character(item_num=1) * GI.up_5star_ep_weapon(item_num=1)
# 导入绘图模块
from GGanalysis.gacha_plot import DrawDistribution
fig = DrawDistribution(dist, dpi=72, show_description=True)
fig.draw_two_graph()
```

更详细的构建抽卡模型和计算分布见项目下的 [example.py](https://github.com/OneBST/GGanalysisLite/blob/main/example.py)

使用绘图程序绘制分为函数见项目下的 [figure_example.py](https://github.com/OneBST/GGanalysisLite/blob/main/figure_example.py)

## 注意事项

目前工具包支持的抽卡层仅适用于满足马尔科夫性质的抽卡模型，即给定现在状态及过去所有状态的情况下，未来抽卡的结果仅仅依赖于当前状态，与过去的状态是独立的。不过好消息是，游戏的抽卡系统基本都满足这样的性质。

当前工具包能实现的抽卡模型是有限的，仅能实现能被给出的四种抽卡层组合出来的模型。对于类似“300井”等，在一定抽数后直接为玩家提供道具的模型，在本工具包框架下仅需简单修改即可。而对于类似不放回抽样的奖品堆模式、集齐碎片兑换模式等，还待之后继续扩展功能。

同时迭代方法计算n连抽获得k个道具概率尚未经过严格数学证明，使用时需要注意。