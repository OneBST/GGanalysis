# GGanalysisLite计算包

这个版本的计算包追求计算速度，而[GGanalysis](https://github.com/OneBST/GGanalysis)包有着更多计算功能。

GGanalysisLite包通过卷积计算分布列，通过FFT和快速幂加速卷积计算。

测试玩家得到的排名值`rankA` `rankB`的数学意义是：与抽了同样数量五星的其他玩家相比，测试玩家花费的抽数大于等于`rankA`比例的玩家，小于等于`rankB`比例的玩家。`rankA` `rankB`的值在(0,1]上，`rankA`值越高说明玩家越倒霉，达到1则说明达到了保底系统的极限。`rankB`值越高说明玩家越幸运，达到1则说明没有更幸运的玩家。

### 安装方法

打开终端输入以下指令，安装完成后下载文件可以删除。

```shell
git clone https://github.com/OneBST/GGanalysisLite.git
cd GGanalysisLite
pip install .
```

### 卷积计算方法

采用的模型见我的[B站专栏](https://www.bilibili.com/read/cv10468091)。通过卷积计算独立同分布且分布有界事件多次重复后的分布列。

卷积采用`scipy`的`signal.convolve()`函数，由其自动根据规模选择使用FFT计算或是直接卷积。

对于一个分布列截止在![](http://latex.codecogs.com/svg.latex?K)且重复试验独立的随机事件，有![](http://latex.codecogs.com/svg.latex?N)次试验叠加时，通过朴素的动态规划计算分布的复杂度是![](http://latex.codecogs.com/svg.latex?O(K^2N^2) )，这样的方法并不够快。如果采用朴素的FFT计算卷积结果，复杂度为![](http://latex.codecogs.com/svg.latex?O(KN^2log_2{KN}))，故采用快速幂进行加速，复杂度降为![](http://latex.codecogs.com/svg.latex?O(KNlog_2{KN}))。一般来说![](http://latex.codecogs.com/svg.latex?K)较小且为常数，此时复杂度为![](http://latex.codecogs.com/svg.latex?O(Nlog_2{N}))

![](http://latex.codecogs.com/svg.latex?{\sum_{n=1}^{log_2N}{n2^n}\textless\sum_{n=1}^{log_2N}{log_2N{\cdot}2^n}\textless2Nlog_2N{\Rightarrow}O(Nlog_2N)})

### 用例

**“非酋度”及“欧皇度”计算**

使用`GenshinPlayer`类添加玩家，初始化其抽卡情况，随后使用类函数输出`rankA` `rankB`两个排名值。（注意这里的排名值只统计到玩家最后一次抽到五星/UP五星位置，垫抽部分和垫出的“大保底”没有考虑，需要更详细的计算请用[GGanalysis](https://github.com/OneBST/GGanalysis)包）

返回的`rankA`是你比多大比例的玩家更倒霉或一样倒霉，称为“非酋度”。`rankB`是你比多大比例的玩家更幸运或者一样幸运，称为“欧皇度”。这两个值都大于0，最高为1。

```python
import GGanalysisLite as ggl

permanent_5 = 5			# 常驻祈愿五星数量
permanent_pull = 301	# 常驻祈愿恰好出最后一个五星时花费的抽数
character_5 = 14		# 角色祈愿五星数量
character_pull = 876	# 角色祈愿恰好出最后一个五星时花费的抽数
character_up_5 = 8		# 角色祈愿UP五星数量
character_up_pull = 876	# 角色祈愿恰好出最后一个UP五星时花费的抽数
weapon_5 = 2			# 武器祈愿五星数量
weapon_pull = 126		# 武器祈愿恰好出最后一个五星时花费的抽数

# 初始化玩家
player = ggl.GenshinPlayer(	p5=permanent_5,
                            c5=character_5,
                            u5=character_up_5,
                            w5=weapon_5,
                            p_pull=permanent_pull,
                            c_pull=character_pull,
                            u_pull=character_up_pull,
                            w_pull=weapon_pull)

# 查看常驻祈愿rank（仅考虑五星数量）
print('常驻祈愿', player.get_p5_rank())

# 查看角色祈愿rank（仅考虑五星数量）
print('角色祈愿', player.get_c5_rank())

# 查看UP角色rank（仅考虑UP五星数量）
print('角色祈愿UP', player.get_u5_rank())

# 查看武器祈愿rank（仅考虑五星数量）
print('武器祈愿', player.get_w5_rank())

# 查看综合rank（角色祈愿仅考虑五星数量，常驻祈愿仅考虑五星数量，武器祈愿仅考虑五星数量）
print('综合', player.get_comprehensive_rank())

# 查看综合rank（角色祈愿考虑UP数量，常驻祈愿仅考虑五星数量，武器祈愿仅考虑五星数量）
print('综合UP', player.get_comprehensive_rank_Up5Character())
```

**条件分布列计算**

玩家经常想知道，在现在的基础上，再抽多少抽可以抽到想要的角色。或是想知道抽n个UP角色和m个定轨五星武器的抽数分布，以了解准备多少抽有多大把握可以抽到。这些的计算也很简单，以下是使用工具包获得分布列的简单实现。在获取了分布列后，简单累加就是概率。

```python
# 计算垫了30抽，有大保底，抽取2个UP五星角色所需抽数的分布列
c = ggl.Up5starCharacter()
dist_c = c.conditional_distribution(2, pull_state=30, up_guarantee=1)

# 计算垫了10抽，有大保底，命定值为0，抽取2个定轨UP五星武器所需抽数的分布列
w = ggl.UP5starWeaponEP()
dist_w = w.conditional_distribution(2, pull_state=10, up_guarantee=1， fate_point=0)

# 计算在没有垫抽情况下获得2个UP角色和1个定轨五星武器的抽数分布列
dist_cw = ggl.GI_conv_cw(3, 0, 0, 1, 0, 0, 0)
```

