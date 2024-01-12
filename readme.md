# 抽卡游戏概率分析工具包-GGanalysis

本工具包为快速构建抽卡游戏抽卡模型设计，通过引入“抽卡层”，以抽卡层的组合快速实现复杂的抽卡逻辑，并以较低的时间复杂度计算抽卡模型对应的分布。除了计算分布，工具包还提供一些将计算结果可视化的绘图工具，并计划加入更多的针对各类抽卡问题的计算工具和设计工具。

近期加入了对有分值道具的相关功能，可以对类似原神圣遗物等问题进行建模。但请注意，这部分代码随时可能发生变动，接口可能大幅改动。

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

画图时需要安装[思源黑体](https://github.com/adobe-fonts/source-han-sans)，安装[对应版本](https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip)后即可使用，若出现找不到字体的情况，Windows 下检查 `C:/Windows/Fonts/` 下是否有以 `SourceHanSansSC` 开头的otf字体（Linux 则检查 `~/.local/share/fonts/`）

如果安装后还是找不到字体，请将 `GGanalysis/plot_tools.py` 内 `mpl.rcParams['font.family'] = 'Source Han Sans SC'` 自行修改为你指定的字体。

## 使用方法

可以在[在线文档](https://onebst.github.io/GGanalysis/)中查看详细使用指南，这里简单举例

**使用定义好的抽卡模型计算抽卡所需抽数分布**

``` python
# 计算抽卡所需抽数分布律 以原神为例
import GGanalysis.games.genshin_impact as GI
# 原神角色池的计算
print('角色池在垫了20抽，有大保底的情况下抽3个UP五星抽数的分布')
dist_c = GI.up_5star_character(item_num=3, item_pity=20, up_pity=1)
print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)

# 计算抽卡所需抽数分布律 以明日方舟为例
import GGanalysis.games.arknights as AK
# 普池双UP的计算 item_num是要抽多少个 item_pity是当前垫了多少抽，从零开始填0就行
dist_c = AK.dual_up_specific_6star(item_num=3, item_pity=20)
print('期望为', dist_c.exp, '方差为', dist_c.var, '分布为', dist_c.dist)
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

每个游戏的抽卡模型定义在 `GGanalysis/games/gamename/gacha_model.py` 文件中

每个游戏的绘图程序在项目 `GGanalysis/games/gamename/figure_plot.py` 文件下可参考

## 注意事项

目前工具包支持的抽卡层仅适用于满足马尔科夫性质的抽卡模型，即给定现在状态及过去所有状态的情况下，未来抽卡的结果仅仅依赖于当前状态，与过去的状态是独立的。不过好消息是，游戏的抽卡系统基本都满足这样的性质。

当前工具包能实现的抽卡模型是有限的，仅能实现能被给出的四种抽卡层组合出来的模型。对于类似“300井”等，在一定抽数后直接为玩家提供道具的模型，在本工具包框架下仅需简单修改即可。而对于类似不放回抽样的奖品堆模式、集齐碎片兑换模式等，还待之后继续扩展功能。

同时迭代方法计算n连抽获得k个道具概率尚未经过严格数学证明，使用时需要注意。

## 参与项目

你可以一起参与本项目的建设，添加更多游戏的支持，提交 pull request 时先选择提交到 `develop` 分支，确定无问题后会被合并到 `main` 分支。

如果要在本地进行开发，不推荐使用以上安装方法，请使用：

``` shell
pip install -e . 
```

这样调用包时会使用本地文件位置的包，就可以随时使用本地更改过的版本了！