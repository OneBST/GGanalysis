import GGanalysis as gg
import GGanalysis.games.blue_archive as BA
from GGanalysis.gacha_plot import QuantileFunction, DrawDistribution
from GGanalysis.basic_models import cdf2dist
from GGanalysis.plot_tools import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import time
import copy

def BA_character(x):
    return str(x)+'个'

def plot_dual_collection_p(title="蔚蓝档案200抽内集齐两个UP学生概率", other_charactors=20, dpi=300, save_fig=False):
    model = BA.SimpleDualCollection(other_charactors=other_charactors)
    both_ratio, a_ratio, b_ratio, none_ratio = model.get_dist(calc_pull=200)
    # 使用fill_between画堆积图
    fig = plt.figure(figsize=(10, 6)) # 27寸4K显示器dpi=163
    ax = plt.gca()
    # 设置x，y范围，更美观
    ax.set_xlim(-5, 205)
    ax.set_ylim(0, 1.05)
    # 开启主次网格和显示及轴名称
    ax.grid(visible=True, which='major', linestyle='-', linewidth=1)
    ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5)
    ax.minorticks_on()
    ax.set_xlabel('抽数', weight='bold', size=12, color='black')
    ax.set_ylabel('累进概率', weight='bold', size=12, color='black')
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    # 绘制右侧坐标轴
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    # ax2.set_ylabel('Y2 data', color='r')  # 设置第二个y轴的标签
    # 绘制图像
    x = range(1, len(both_ratio))
    plt.fill_between(x, 0, both_ratio[1:], label='集齐AB', color='C0', alpha=0.7, edgecolor='none', zorder=10)
    plt.fill_between(x, both_ratio[1:], both_ratio[1:]+a_ratio[1:], label='只获得A类', color='C0', alpha=0.5, edgecolor='none', zorder=10)
    plt.fill_between(x, both_ratio[1:]+a_ratio[1:], both_ratio[1:]+a_ratio[1:]+b_ratio[1:], label='只获得B类', color='C0', alpha=0.3, edgecolor='none', zorder=10)
    plt.plot(x, both_ratio[1:], color='C0', alpha=0.7, linewidth=2, zorder=10)
    plt.plot(x, both_ratio[1:]+a_ratio[1:], color='C0', alpha=0.5, linewidth=2, zorder=10)
    plt.plot(x, both_ratio[1:]+a_ratio[1:]+b_ratio[1:], color='C0', alpha=0.3, linewidth=2, zorder=10)
    # 添加来回切换策略
    swith_cdf = BA.no_exchange_dp()
    plt.plot(
        range(1, len(swith_cdf)), swith_cdf[1:], color='gold', alpha=1, linewidth=2, zorder=10, linestyle='--', label='对比策略')
    # 添加描述文本
    ax.text(
        30, 1.025,
        f"采用国服官方公示模型 A为本池UP角色 B为同期另一池角色，填充部分为堆积概率\n考虑常驻角色会越来越多，以除去UP角色外卡池内有{other_charactors}个角色的情况进行计算\n"+
        f"对比策略指一直单抽，若在卡池A中抽到A，则换池抽B，无兑换下集齐概率\n@一棵平衡树 "+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        weight='bold',
        size=12,
        color='#B0B0B0',
        path_effects=stroke_white,
        horizontalalignment='left',
        verticalalignment='top'
    )
    # 图例和标题
    plt.legend(loc='upper left', prop={'weight': 'normal'})
    plt.title(title, weight='bold', size=18)
    if save_fig:
        plt.savefig(os.path.join('figure', title+'.png'), dpi=dpi)
    else:
        plt.show()

def get_dual_description(
        item_name='道具',
        cost_name='抽',
        text_head=None,
        mark_exp=None,
        direct_exchange=None,
        show_max_pull=None,
        is_finite=None,
        text_tail=None
    ):
    '''用于描述蔚蓝航线同时得到两个同时UP的角色'''
    description_text = ''
    # 开头附加文字
    if text_head is not None:
        description_text += text_head
    # 对道具期望值的描述
    if mark_exp is not None:
        if description_text != '':
            description_text += '\n'
        description_text += '集齐同时UP角色的期望抽数为'+format(mark_exp, '.2f')+cost_name
    # 对能否100%获取道具的描述
    description_text += '\n集齐同时UP角色最多需要400抽'
    # 末尾附加文字
    if text_tail is not None:
        if description_text != '':
            description_text += '\n'
        description_text += text_tail
    description_text =  description_text.rstrip()
    return description_text

# 神明文字返还
def plot_refund(title="蔚蓝档案招募神名文字返还", dots=100, dpi=300, save_fig=False):
    ans_base = np.zeros(dots+1, dtype=float)
    for i in range(dots+1):
        ans_base[i] = BA.get_common_refund(rc3=i/dots)
    ans_high = np.zeros(dots+1, dtype=float)
    for i in range(dots+1):
        ans_high[i] = BA.get_common_refund(rc3=i/dots, has_up=True)
    # 使用fill_between画堆积图
    fig = plt.figure(figsize=(10, 6)) # 27寸4K显示器dpi=163
    ax = plt.gca()
    # 设置x，y范围，更美观
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 5.15)
    # 开启主次网格和显示及轴名称
    ax.grid(visible=True, which='major', linestyle='-', linewidth=1)
    ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5)
    ax.minorticks_on()
    ax.set_xlabel('常驻3星学生集齐比例', weight='bold', size=12, color='black')
    ax.set_ylabel('每抽获取神名文字期望数量', weight='bold', size=12, color='black')
    plt.xticks(np.arange(0, 1.1, 0.1))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

    # 绘制图像
    plt.fill_between(np.linspace(0, 1, dots+1), 0, ans_base, color='C0', alpha=0.3, edgecolor='none', zorder=10)
    plt.plot(np.linspace(0, 1, dots+1), ans_base, color='C0', alpha=1, linewidth=2, label='初始无UP', zorder=10)

    plt.plot(np.linspace(0, 1, dots+1), ans_high, color='C0', alpha=1, linestyle=':', linewidth=2, label='初始有UP', zorder=10)
    # 添加描述文本
    ax.text(
        30, 1.025,
        f"采用国服官方公示模型 \n不考虑每200抽兑换\n"+
        f"@一棵平衡树 "+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        weight='bold',
        size=12,
        color='#B0B0B0',
        path_effects=stroke_white,
        horizontalalignment='left',
        verticalalignment='top'
    )
    # 图例和标题
    plt.legend(loc='upper left', prop={'weight': 'normal'})
    plt.title(title, weight='bold', size=18)
    if save_fig:
        plt.savefig(os.path.join('figure', title+'.png'), dpi=dpi)
    else:
        plt.show()

if __name__ == '__main__':
    # 传统方法分析UP学生抽取，3星升到最高级需要220学生神名文字，重复返还100，所以最多需要抽4个
    BA_up3dist = [gg.FiniteDist()]
    for i in range(1, 4+1):
        BA_up3dist.append(BA.up_3star(i))
    BA_fig = QuantileFunction(
            BA_up3dist,
            title='蔚蓝档案获取UP三星学生概率(考虑200井)',
            item_name='特定UP三星角色',
            text_head=f'招募内特定UP三星学生概率为{BA.P_3UP:.1%}\n考虑每200抽的招募点数兑换\n不考虑神名文字直接兑换学生神名文字',
            text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
            max_pull=850,
            mark_func=BA_character,
            line_colors=cm.YlOrBr(np.linspace(0.4, 0.9, 4+1)),
            direct_exchange=200,
            y2x_base=2,

            plot_direct_exchange=True,
            is_finite=False)
    BA_fig.show_figure(dpi=300, savefig=True)
    
    # 传统方法分析UP学生抽取，2星升到最高级需要300学生神名文字，重复返还20，所以最多需要抽15个
    BA_up2dist = [gg.FiniteDist()]
    for i in range(1, 15+1):
        BA_up2dist.append(BA.up_2star(i))
    BA_fig = QuantileFunction(
            BA_up2dist,
            title='蔚蓝档案获取UP两星学生概率',
            item_name='特定UP两星角色',
            text_head=f'招募内特定UP两星学生概率为{BA.P_2UP:.1%}\n不考虑神名文字直接兑换学生神名文字',
            text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
            max_pull=950,
            mark_func=BA_character,
            line_colors=cm.GnBu(np.linspace(0.4, 0.9, 15+1)),
            y2x_base=2,

            plot_direct_exchange=True,
            is_finite=False)
    BA_fig.show_figure(dpi=300, savefig=True)

    # 分析无保底时同时集齐UP的两个学生的概率(考虑换池抽)
    plot_dual_collection_p(dpi=300, save_fig=True)

    # 分析同时集齐UP的两个学生的抽数分布(不考虑换池抽，对于抽到200井的情况两种策略对3星是无区别的)
    # 集齐两个UP角色
    model = BA.SimpleDualCollection(other_charactors=BA.STANDER_3STAR)
    both_ratio, a_ratio, b_ratio, none_ratio = model.get_dist(calc_pull=400)
    temp_dist = copy.deepcopy(both_ratio)
    temp_dist[200:] += a_ratio[200:] + b_ratio[200:]
    temp_dist[400] = 1
    temp_dist = cdf2dist(temp_dist)
    BA_fig = DrawDistribution(
        dist_data=temp_dist,
        title='蔚蓝档案集齐同时UP的两个学生',
        quantile_pos=[0.05, 0.1, 0.75, 0.8, 0.9, 0.95, 1],
        max_pull=400,
        text_head=f'采用官方公示模型\n视常驻有{BA.STANDER_3STAR}个角色，计入每{BA.EXCHANGE_PULL}抽兑换',
        text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
        description_pos=0,
        is_finite=True,
        description_func=get_dual_description,
    )
    BA_fig.show_two_graph(dpi=300, savefig=True)

    # 计算神名文字返还期望
    plot_refund(save_fig=True)
    
    
