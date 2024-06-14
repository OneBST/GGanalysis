import GGanalysis.games.wuthering_waves as WW
from GGanalysis.gacha_plot import QuantileFunction, DrawDistribution
from GGanalysis import FiniteDist
import matplotlib.cm as cm
import numpy as np
import time

def WW_item(x):
    return '抽取'+str(x)+'个'

# 鸣潮5星物品分布
WW_fig = DrawDistribution(
    dist_data=WW.common_5star(1),
    title='鸣潮获取五星物品抽数分布',
    text_head='采用推测模型，不保证完全准确',
    text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    item_name='五星物品',
    quantile_pos=[0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.99],
    is_finite=True,
)
WW_fig.show_dist(dpi=300, savefig=True)

# 鸣潮获取UP5星物品
WW_fig = QuantileFunction(
    WW.common_5star(7, multi_dist=True),
    title='鸣潮获取五星道具概率',
    item_name='五星道具',
    text_head='采用推测模型，不保证完全准确\n获取1个五星道具最多80抽',
    text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    max_pull=575,
    line_colors=cm.Oranges(np.linspace(0.5, 0.9, 7+1)),
    # y_base_gap=25,
    # y2x_base=2,
    is_finite=True)
WW_fig.show_figure(dpi=300, savefig=True)

# 鸣潮获取UP5星角色
WW_fig = QuantileFunction(
    WW.up_5star_character(7, multi_dist=True),
    title='鸣潮获取UP五星角色概率',
    item_name='UP五星角色',
    text_head='采用推测模型，不保证完全准确\n获取1个UP五星角色最多160抽',
    text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    max_pull=1150,
    line_colors=cm.YlOrBr(np.linspace(0.4, 0.9, 7+1)),  # cm.OrAKges(np.linspace(0.5, 0.9, 6+1)),
    y_base_gap=25,
    y2x_base=2,
    is_finite=True)
WW_fig.show_figure(dpi=300, savefig=True)