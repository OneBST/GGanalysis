# 抽卡游戏概率分析工具包-GGanalysisLite

本工具包最开始用于原神抽卡概率计算，想了一下为什么不扩展到更多游戏呢，于是重构了一下代码加入了对更多抽卡模型的支持。GGanalysisLite里的GG以前是指Genshin Impact，现在可以指Gacha Game了。

这个版本的计算包追求计算速度，而[GGanalysis](https://github.com/OneBST/GGanalysis)包有着更多计算功能（仅限于计算原神抽卡概率）。

GGanalysisLite包通过卷积计算分布列，通过FFT和快速幂加速卷积计算。

## 安装方法

打开终端输入以下指令，安装完成后下载文件可以删除。

```shell
git clone https://github.com/OneBST/GGanalysisLite.git
cd GGanalysisLite
pip install .
```

## 支持抽卡层

1. `Pity_layer` 保底抽卡层
2. `Bernoulli_layer` 伯努利抽卡层
3. `Markov_layer` 马尔可夫抽卡层
3. `Coupon_Collector_layer` 集齐道具层（没有完工，还不完善）



## 使用方法

见项目下的 [example.py](https://github.com/OneBST/GGanalysisLite/blob/main/example.py)