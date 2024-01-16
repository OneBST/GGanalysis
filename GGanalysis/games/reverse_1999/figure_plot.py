import GGanalysis.games.reverse_1999 as RV
from GGanalysis.gacha_plot import QuantileFunction, DrawDistribution
from GGanalysis import FiniteDist
import matplotlib.cm as cm
import numpy as np
import time

def RV_character(x):
    return '塑造'+str(x-1)
def RV_collection(x):
    return '集齐'+str(x)+'种'

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

# 重返未来1999 常驻集齐UP6星角色
ans_list = [FiniteDist()]
for i in range(1, 12):
    ans_list.append(RV.stander_charactor_collection(target_types=i))
RV_fig = QuantileFunction(
        ans_list,
        title='重返未来1999集齐常驻6星角色概率',
        item_name='常驻6星角色',
        text_head='采用官方公示模型',
        text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        max_pull=3800,
        mark_func=RV_collection,
        line_colors=cm.PuRd(np.linspace(0.2, 0.9, 11+1)),
        y_base_gap=25,
        mark_offset=-0.4,
        y2x_base=3,
        mark_exp=False,
        is_finite=False)
RV_fig.show_figure(dpi=300, savefig=True)

# 重返未来1999 集齐两个UP五星角色
RV_fig = DrawDistribution(
    dist_data=RV.both_up_5star(),
    title='重返未来1999集齐两个UP五星角色',
    max_pull=300,
    text_head='采用官方公示模型',
    text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    description_pos=200,
    is_finite=False,
)
RV_fig.show_dist(dpi=300, savefig=True)

# 重返未来1999 获取六星角色
RV_fig = DrawDistribution(
    dist_data=RV.common_6star(1),
    title='重返未来1999获取六星角色',
    text_head='采用官方公示模型',
    text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    is_finite=True,
)
RV_fig.show_dist(dpi=300, savefig=True)

# 重返未来1999 常驻获取特定六星角色
RV_fig = DrawDistribution(
    dist_data=RV.specific_stander_6star(1),
    title='重返未来1999常驻获取特定六星角色',
    text_head='采用官方公示模型',
    text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    max_pull=2300,
    description_pos=1550,
    quantile_pos=[0.25, 0.5, 0.75, 0.9, 0.99],
    is_finite=False,
)
RV_fig.show_dist(dpi=300, savefig=True)

# 重返未来1999 获取UP六星角色
RV_fig = DrawDistribution(
    dist_data=RV.up_6star(1),
    title='重返未来1999获取UP六星角色',
    text_head='采用官方公示模型',
    text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    is_finite=True,
)
RV_fig.show_dist(dpi=300, savefig=True)

# 重返未来1999 新春限定池获取UP六星角色
RV_fig = QuantileFunction(
    RV.up_6star(6, multi_dist=True),
    title='重返未来1999新春限定六星角色概率（200井）',
    item_name='UP六星角色',
    text_head='采用官方公示模型\n获取1个UP六星角色最多140抽\n考虑全部兑换六星角色\n最多560抽满塑造',
    text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    max_pull=560,
    mark_func=RV_character,
    line_colors=cm.YlOrBr(np.linspace(0.1, 0.95, 6+1)),  # cm.OrAKges(np.linspace(0.5, 0.9, 6+1)),
    direct_exchange=200,
    y_base_gap=25,
    y2x_base=3,
    plot_direct_exchange=True,
    is_finite=False)
RV_fig.show_figure(dpi=300, savefig=True)

# 重返未来1999 自选双常驻六星池获取特定六星角色
RV_fig = QuantileFunction(
    RV.dual_up_specific_6star(6, multi_dist=True),
    title='重返未来1999自选双常驻六星池获取特定六星角色',
    item_name='特定六星角色',
    text_head='采用官方公示模型\n获取1个任意选定六星角色最多140抽',
    text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    max_pull=1400,
    mark_func=RV_character,
    line_colors=cm.YlOrBr(np.linspace(0.4, 0.9, 6+1)),  # cm.OrAKges(np.linspace(0.5, 0.9, 6+1)),
    y_base_gap=25,
    is_finite=False)
RV_fig.show_figure(dpi=300, savefig=True)

def collection_description(
        item_name='角色',
        cost_name='抽',
        text_head=None,
        mark_exp=None,
        direct_exchange=None,
        show_max_pull=None,
        is_finite=None,
        text_tail=None
    ):
    description_text = ''
    # 开头附加文字
    if text_head is not None:
        description_text += text_head
    # 对道具期望值的描述
    if mark_exp is not None:
        if description_text != '':
            description_text += '\n'
        description_text += '集齐两个自选'+item_name+'期望为'+format(mark_exp, '.2f')+cost_name
    # 末尾附加文字
    if text_tail is not None:
        if description_text != '':
            description_text += '\n'
        description_text += text_tail
    description_text =  description_text.rstrip()
    return description_text

# 重返未来1999 自选双常驻六星池集齐自选六星角色
RV_fig = DrawDistribution(
    RV.dual_up_both_6star(),
    title='重返未来1999自选双常驻六星池集齐自选六星角色',
    text_tail='无法确保在有限抽数内一定集齐\n@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    is_finite=False,
    max_pull=550,
    show_peak=False,
    description_func=collection_description,
)
RV_fig.show_two_graph(dpi=300, savefig=True)