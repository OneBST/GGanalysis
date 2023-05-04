## 采用模型

工具包里原神模块基于的抽卡模型见[原神抽卡全机制总结](https://bbs.nga.cn/read.php?tid=26754637)，是非常准确的模型。为了实现方便，工具包中对UP四星角色、UP四星武器、UP五星武器时不考虑从常驻中歪到的情况，计算四星物品时忽略四星物品被五星物品顶到下一抽的情况。这些近似实际影响很小。

绘制概率图表见[原神抽卡概率工具表](https://bbs.nga.cn/read.php?tid=28026734)

## 其他

写了一个估算总氪金数的小程序`GetCost.py`，比较粗糙而且没怎么检查，可以用着玩一下，以后会仔细想想放哪里。

还有一个在测试的`PredictNextType.py`用于计算普池下一个五星是角色还是武器，会给出概率，还在观察有无BUG，也是可以玩一下的玩意。