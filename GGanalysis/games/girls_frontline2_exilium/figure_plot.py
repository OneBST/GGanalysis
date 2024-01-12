import GGanalysis.games.girls_frontline2_exilium as GF2
from GGanalysis.gacha_plot import QuantileFunction
import matplotlib.cm as cm
import numpy as np
import time

# 绘制少女前线2：追放抽卡概率分位图表

# 定义获取物品个数的别名
def gf2_character(x):
    return str(x-1)+'锥'
def gf2_weapon(x):
    return str(x)+'校'

# 少前2追放UP精英角色
GF2_fig = QuantileFunction(
        GF2.up_elite(item_num=7, multi_dist=True),
        title='少前2追放UP精英人形抽取概率',
        item_name='UP精英人形',
        text_head='本算例中UP物品均不在常驻祈愿中',
        text_tail='采用GGanalysis库绘制 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=900,
        line_colors=cm.Oranges(np.linspace(0.3, 0.9, 7+1)),
        mark_func=gf2_character,
        is_finite=True)
GF2_fig.show_figure(dpi=300, savefig=True)

# 少前2追放特定UP标准角色
GF2_fig = QuantileFunction(
        GF2.up_common_specific_character(item_num=7, multi_dist=True),
        title='少前2追放特定UP标准人形抽取概率',
        item_name='特定UP标准人形',
        text_head='本算例中UP物品均不在常驻祈愿中\n绘图曲线忽略精英与标准耦合情况',
        text_tail='采用GGanalysis库绘制 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=1200,
        mark_func=gf2_character,
        line_colors=cm.Purples(np.linspace(0.5, 0.9, 7+1)),
        is_finite=False)
GF2_fig.show_figure(dpi=300, savefig=True)

# 少前2追放定轨UP精英武器
GF2_fig = QuantileFunction(
        GF2.up_elite_weapon(item_num=6, multi_dist=True),
        title='少前2追放UP精英武器抽取概率',
        item_name='UP精英武器',
        text_head='本算例中UP物品均不在常驻祈愿中',
        text_tail='采用GGanalysis库绘制 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=600,
        mark_func=gf2_weapon,
        line_colors=cm.Reds(np.linspace(0.5, 0.9, 6+1)),
        is_finite=True)
GF2_fig.show_figure(dpi=300, savefig=True)

# 少前2追放特定UP标准武器
GF2_fig = QuantileFunction(
        GF2.up_common_specific_weapon(item_num=6, multi_dist=True),
        title='少前2追放特定UP标准武器抽取概率',
        item_name='特定UP标准武器',
        text_head='本算例中UP物品均不在常驻祈愿中\n绘图曲线忽略精英与标准耦合情况',
        text_tail='采用GGanalysis库绘制 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=500,
        mark_func=gf2_weapon,
        is_finite=False)
GF2_fig.show_figure(dpi=300, savefig=True)