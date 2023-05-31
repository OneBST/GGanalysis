import GGanalysis.games.reverse_1999 as RV
from GGanalysis.gacha_plot import QuantileFunction, DrawDistribution
from GGanalysis import FiniteDist
import matplotlib.cm as cm
import numpy as np
import time

def RV_character(x):
    return '塑造'+str(x-1)

# 重返未来1999 UP6星角色
RV_fig = QuantileFunction(
        RV.up_6star(6, multi_dist=True),
        title='重返未来1999 UP六星角色抽取概率',
        item_name='UP六星角色',
        text_head='采用官方公示模型\n获取1个UP六星角色最多140抽',
        text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=750,
        mark_func=RV_character,
        line_colors=cm.YlOrBr(np.linspace(0.4, 0.9, 6+1)),  # cm.OrAKges(np.linspace(0.5, 0.9, 6+1)),
        y_base_gap=25,
        is_finite=None)
RV_fig.show_figure(dpi=300, savefig=True)

# 重返未来1999 UP5星角色
ans_list = [FiniteDist()]
for i in range(1, 7):
    ans_list.append(RV.specific_up_5star(i))
RV_fig = QuantileFunction(
        ans_list,
        title='重返未来1999 特定UP五星角色抽取概率',
        item_name='UP五星角色',
        text_head='采用官方公示模型',
        text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=750,
        mark_func=RV_character,
        line_colors=cm.GnBu(np.linspace(0.4, 0.9, 6+1)),  # cm.OrAKges(np.linspace(0.5, 0.9, 6+1)),
        y_base_gap=25,
        is_finite=False)
RV_fig.show_figure(dpi=300, savefig=True)

# 重返未来1999 集齐两个UP五星角色
# 需要将 self.show_description 的横坐标改为 210
RV_fig = DrawDistribution(
    dist_data=RV.both_up_5star(),
    max_pull=300,
    title='重返未来1999集齐两个UP五星角色概率',
    is_finite=False,
)
RV_fig.draw_two_graph(dpi=300, savefig=True)

# 重返未来1999 获取六星角色
RV_fig = DrawDistribution(
    dist_data=RV.common_6star(1),
    title='重返未来1999获取六星角色',
    is_finite=True,
)
RV_fig.draw_two_graph(dpi=300, savefig=True)

# 重返未来1999 获取UP六星角色
RV_fig = DrawDistribution(
    dist_data=RV.up_6star(1),
    title='重返未来1999获取UP六星角色',
    is_finite=True,
)
RV_fig.draw_two_graph(dpi=300, savefig=True)