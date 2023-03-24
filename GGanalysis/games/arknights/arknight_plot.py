import GGanalysis.games.arknights as AK
from GGanalysis.gacha_plot import QuantileFunction
import matplotlib.cm as cm
import numpy as np
import time

def AK_character(x):
    return '潜能'+str(x)

# 明日方舟6星角色
AK_fig = QuantileFunction(
        AK.common_6star(10, multi_dist=True),
        title='明日方舟六星角色抽取概率',
        item_name='六星角色',
        text_head='请注意试绘图使用模型未必准确',
        text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=600,
        line_colors=0.5*(cm.Blues(np.linspace(0.1, 1, 10+1))+cm.Greys(np.linspace(0.4, 1, 10+1))),
        y2x_base=1.6,
        is_finite=True)
AK_fig.show_figure(dpi=300, savefig=True)

# 明日方舟普池单UP6星角色
AK_fig = QuantileFunction(
        AK.single_up_6star(6, multi_dist=True),
        title='明日方舟普池单UP六星角色抽取概率',
        item_name='UP角色(定向选调)',
        text_head='UP角色占六星中50%概率\n请注意试绘图使用模型未必准确\n考虑定向选调机制(类型保底)\n获取第1个UP角色最多300抽\n无法确保在有限抽数内一定满潜',
        text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=1000,
        mark_func=AK_character,
        line_colors=cm.GnBu(np.linspace(0.3, 0.9, 6+1)),  # cm.OrAKges(np.linspace(0.5, 0.9, 6+1)),
        y_base_gap=25,
        is_finite=None)
AK_fig.show_figure(dpi=300, savefig=True)

# 明日方舟普池双UP6星角色
AK_fig = QuantileFunction(
        AK.dual_up_specific_6star(6, multi_dist=True),
        title='明日方舟普池双UP获取特定六星角色概率',
        item_name='特定UP角色(第1个)',
        text_head='两个角色各占六星中25%概率\n请注意试绘图使用模型未必准确\n获取第1个UP角色最多501抽\n无法确保在有限抽数内一定满潜',
        text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=2200,
        mark_func=AK_character,
        line_colors=cm.PuRd(np.linspace(0.2, 0.9, 6+1)),  # cm.OrAKges(np.linspace(0.5, 0.9, 6+1)),
        y2x_base=1.8,
        is_finite=None)
AK_fig.show_figure(dpi=300, savefig=True)

# 明日方舟限定UP6星角色
AK_fig = QuantileFunction(
        AK.limited_up_6star(6, multi_dist=True),
        title='明日方舟获取限定六星角色概率(考虑300井)',
        item_name='限定六星角色',
        text_head='两个角色各占六星中35%概率\n请注意试绘图使用模型未必准确',
        text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=1100,
        mark_func=AK_character,
        line_colors=cm.YlOrBr(np.linspace(0.4, 0.9, 6+1)),  # cm.OrAKges(np.linspace(0.5, 0.9, 6+1)),
        direct_exchange=300,
        y_base_gap=25,
        y2x_base=1.8,
        plot_direct_exchange=True,
        is_finite=False)
AK_fig.show_figure(dpi=300, savefig=True)

# 明日方舟限定UP6星角色
AK_fig = QuantileFunction(
        AK.limited_up_6star(6, multi_dist=True),
        title='明日方舟获取限定六星角色概率(不考虑300井)',
        item_name='限定六星角色',
        text_head='两个角色各占六星中35%概率\n请注意试绘图使用模型未必准确',
        text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=1400,
        mark_func=AK_character,
        line_colors=cm.YlOrBr(np.linspace(0.4, 0.9, 6+1)),  # cm.OrAKges(np.linspace(0.5, 0.9, 6+1)),
        y_base_gap=25,
        y2x_base=1.8,
        is_finite=False)
AK_fig.show_figure(dpi=300, savefig=True)
