from GGanalysis.basic_models import *

__all__ = [
    'model_1_1_3',
    'model_0_3_2',
    'model_0_2_3',
    'model_0_2_2',
    'model_0_2_1',
    'model_0_1_1',
]

# 按卡池 彩 金 紫 道具数量进行模型命名
model_1_1_3 = GeneralCouponCollectorModel([0.012, 0.02, 0.025/3, 0.025/3, 0.025/3], ['UR1', 'SSR1', 'SR1', 'SR2', 'SR3'])
model_0_3_2 = GeneralCouponCollectorModel([0.02/3, 0.02/3, 0.02/3, 0.025/2, 0.025/2], ['SSR1', 'SSR2', 'SSR3', 'SR1', 'SR2'])
model_0_2_3 = GeneralCouponCollectorModel([0.02/2, 0.02/2, 0.025/3, 0.025/3, 0.025/3], ['SSR1', 'SSR2', 'SR1', 'SR2', 'SR3'])
model_0_2_2 = GeneralCouponCollectorModel([0.02/2, 0.02/2, 0.025/2, 0.025/2], ['SSR1', 'SSR2', 'SR1', 'SR2'])
model_0_2_1 = GeneralCouponCollectorModel([0.02/2, 0.02/2, 0.025], ['SSR1', 'SSR2', 'SR1'])
model_0_1_1 = GeneralCouponCollectorModel([0.02, 0.025], ['SSR1', 'SR1'])
        
if __name__ == '__main__':
    # print(model_1_1_1().exp)
    import GGanalysis as gg
    from matplotlib import pyplot as plt
    # 对于 1_1_1 带200天井的情况，需要单独处理
    dist_0 = model_1_1_3(target_item=['UR1', 'SSR1'])  # 未触发200天井时
    dist_1 = model_1_1_3(target_item=['SSR1'])  # 触发200天井时
    cdf_0 = gg.dist2cdf(dist_0)
    cdf_1 = gg.dist2cdf(dist_1)
    cdf_0 = cdf_0[:min(len(cdf_0), len(cdf_1))]
    cdf_1 = cdf_1[:min(len(cdf_0), len(cdf_1))]
    cdf_0[200:] = cdf_1[200:]
    plt.plot(cdf_0)
    plt.show()    
    # dist_ans = cdf2dist(cdf_0)

