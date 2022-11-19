# 抽卡游戏概率分析工具包-GGanalysis

本工具包为快速构建抽卡游戏抽卡模型设计，通过引入“抽卡层”，以抽卡层的组合快速实现复杂的抽卡逻辑，并以较低的时间复杂度计算抽卡模型对应的分布。除了计算分布，工具包还提供一些将计算结果可视化的绘图工具，并计划加入更多的针对各类抽卡问题的计算工具和设计工具。

工具包名称从GGanalysisLite改为GGanalysis，此前的GGanalysis包暂更名为GGanalysislib。

## 安装方法

本工具包的依赖库很简单，只需要安装`numpy`和`scipy`即可。如果需要使用工具包提供的画图代码，还需安装`matplotlib`。

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

更详细的构建抽卡模型和计算分布见项目下的 [example.py](https://github.com/OneBST/GGanalysisLite/blob/main/example.py)

使用绘图程序绘制分为函数见项目下的 [figure_example.py](https://github.com/OneBST/GGanalysisLite/blob/main/figure_example.py)

## 注意事项

目前工具包支持的抽卡层仅适用于满足马尔科夫性质的抽卡模型，即给定现在状态及过去所有状态的情况下，未来抽卡的结果仅仅依赖于当前状态，与过去的状态是独立的。不过好消息是，游戏的抽卡系统基本都满足这样的性质。

当前工具包能实现的抽卡模型是有限的，仅能实现能被给出的四种抽卡层组合出来的模型。对于类似“300井”等，在一定抽数后直接为玩家提供道具的模型，在本工具包框架下仅需简单修改即可。而对于类似不放回抽样的奖品堆模式、集齐碎片兑换模式等，还待之后继续扩展功能。