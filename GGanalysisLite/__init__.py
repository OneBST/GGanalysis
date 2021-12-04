'''
    原神抽卡概率计算工具包 GGanalysisLite
    by 一棵平衡树OneBST
    抽卡模型参数采用 https://www.bilibili.com/read/cv10468091
    本计算包计算的情况较为受限，如需精细计算请使用 GGanalysis包
    计算包采用卷积方法计算分布，通过FFT+快速幂加速
'''
from GGanalysisLite.GachaType import *
from GGanalysisLite.ConvDist import *