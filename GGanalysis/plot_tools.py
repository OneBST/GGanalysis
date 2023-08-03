import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.transforms as transforms

mpl.rcParams['font.family'] = 'Source Han Sans SC'

# 描边预设
stroke_white = [pe.withStroke(linewidth=2, foreground='white')]
stroke_black = [pe.withStroke(linewidth=2, foreground='black')]

# matplotlib 风格预设
FIG_PRESET = {
    'figure.dpi':163.18/2,
    'axes.linewidth':1,
    'grid.color':'lightgray',
}

# 初始化fig，返回没有绘制线条但添加了方形网格线的fig和ax
@mpl.rc_context(FIG_PRESET)
def set_square_grid_fig(
        max_pull,           # 绘图时的最高抽数
        x_grids=10,         # x方向网格个数
        y_base_gap=50,      # y轴刻度基本间隔
        y2x_base=4/3,       # y对x的宽高比
        y_force_gap=None,   # 强制设置y间隔，默认为空
    ):
    # 设置绘图大小，固定横向大小，动态纵向大小
    graph_x_space = 5       # axes横向的大小
    x_pad = 1.2             # 横向两侧留空大小
    y_pad = 1.2             # 纵向两侧留空大小
    title_y_space = 0.1     # 为标题预留大小

    # 设置刻度间隔
    x_gap = 1 / x_grids     # x方向刻度间隔
    # 简单的自动确定 y_gap y刻度间隔为base_gap的整数倍
    y_gap = y_base_gap * math.ceil((max_pull / ((x_grids+1) * max(y2x_base, math.log(max_pull)/5) - 1) / y_base_gap))
    if y_force_gap is not None:
        y_gap = y_force_gap
    # 自动确定y方向网格个数
    y_grids = math.ceil(max_pull / y_gap)
    graph_y_space = (y_grids + 1) * graph_x_space / (x_grids + 1)  # 确定axes纵向大小

    # 确定figure大小
    x_size = graph_x_space+x_pad
    y_size = graph_y_space+title_y_space+y_pad
    fig_size = [x_size, y_size]
    fig = plt.figure(figsize=fig_size) # 27寸4K显示器dpi=163
    
    # 创建坐标轴
    ax = fig.add_axes([0.7*x_pad/x_size, 0.6*y_pad/y_size, graph_x_space/x_size, graph_y_space/y_size])
    # 设置坐标轴格式，横轴为百分比显示
    ax.set_xticks(np.arange(0, 1.01, x_gap))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_yticks(np.arange(0, (y_grids+2)*y_gap, y_gap))
    # 设置x，y范围，更美观
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.4*y_gap, (y_grids+0.4)*y_gap)
    # 开启主次网格和显示
    ax.grid(visible=True, which='major', linestyle='-', linewidth=1)
    ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5)
    ax.minorticks_on()

    return fig, ax, x_gap, x_grids, y_gap, y_grids

def add_stroke_dot(ax, x, y, path_effects=stroke_white, zorder=10, *args, **kwargs):
    '''增加带描边的点'''
    ax.scatter(
        x, y, zorder=zorder,
        path_effects=path_effects,
        *args, **kwargs
    ) 
    return ax

# 默认道具获取名称
def default_item_num_mark(x):
    return str(x)+'个'

# 默认显示文字函数
def get_default_description(
        item_name='道具',
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
        description_text += '获取一个'+item_name+'期望为'+format(mark_exp, '.2f')+cost_name
        if direct_exchange is not None:
            description_text += '\n每'+str(direct_exchange)+cost_name+'可额外兑换'+item_name+'\n含兑换期望为'+format(1/(1/mark_exp+1/direct_exchange), '.2f')+cost_name
    # 对能否100%获取道具的描述
    if show_max_pull is not None:
        if is_finite is None:
            pass
        elif is_finite:
            if direct_exchange is None:
                if description_text != '':
                    description_text += '\n'
                description_text += '获取一个'+item_name+'最多需要'+str(show_max_pull)+cost_name
        else:
            if description_text != '':
                description_text += '\n'
            description_text += '无法确保在有限抽数内一定获得'+item_name
    # 末尾附加文字
    if text_tail is not None:
        if description_text != '':
            description_text += '\n'
        description_text += text_tail
    description_text =  description_text.rstrip()
    return description_text

# 增加分位函数
def add_quantile_line(
        ax,
        data: np.array,
        color='C0',
        linewidth=2.5,
        max_pull=None,
        add_end_mark=False,
        is_finite=True,
        quantile_pos: list=None,
        item_num=1,
        add_vertical_line=False,
        y_gap=50,
        mark_pos=0.5,
        text_bias_x=-3/100,
        text_bias_y=3/100,
        mark_offset=-0.3,
        path_effects=stroke_white,
        mark_func=default_item_num_mark,
        *args, **kwargs
    ):
    '''按垂直方向绘制 cdf 曲线'''
    if max_pull is None:
        max_pull = len(data)
    # 图线绘制
    ax.plot(data[:max_pull],
            range(max_pull),
            linewidth=linewidth,
            color=color,
            *args, **kwargs
            )
    # 分位点绘制
    if quantile_pos is not None:
        offset_1 = transforms.ScaledTranslation(text_bias_x, text_bias_y, plt.gcf().dpi_scale_trans)
        offset_2 = transforms.ScaledTranslation(mark_offset+text_bias_x, text_bias_y, plt.gcf().dpi_scale_trans)
        offset_3 = transforms.ScaledTranslation(1.3*text_bias_x, text_bias_y, plt.gcf().dpi_scale_trans)
        transform_1 = ax.transData + offset_1
        transform_2 = ax.transData + offset_2
        transform_3 = ax.transData + offset_3
        # 分位点及分位文字绘制
        for p in quantile_pos:
            pos = np.searchsorted(data, p, side='left')
            if pos >= len(data):
                continue
            # 插值算出一个位置
            dot_y = (p-data[pos-1])/(data[pos]-data[pos-1])+pos-1
            add_stroke_dot(ax, p, dot_y, s=3, color=color)
            # 增加抽数信息
            ax.text(p, dot_y, str(pos),
                    weight='medium',
                    size=12,
                    color='black',
                    transform=transform_1,
                    horizontalalignment='right',
                    path_effects=path_effects
                )
            # 在设置位置标注说明文字
            if p == mark_pos:
                plt.text(p, dot_y, mark_func(item_num),
                    weight='bold',
                    size=12,
                    color=color,
                    transform=transform_2,
                    horizontalalignment='right',
                    path_effects=path_effects
                )
            # 在标记概率绘制竖虚线
            if add_vertical_line:
                ax.plot([p, p], [-y_gap/2, dot_y+y_gap*2], c='gray', linewidth=2, linestyle=':')
                ax.text(p, dot_y+y_gap*1.5, str(int(p*100))+"%",
                    transform=transform_3,
                    color='gray',
                    weight='bold',
                    size=12,
                    horizontalalignment='right',
                    path_effects=path_effects
                )
    if add_end_mark:
        # 有界分布
        if is_finite:
            add_stroke_dot(ax, 1, len(data)-1, s=10, color=color, marker="o")
        # 长尾分布
        else:
            offset = transforms.ScaledTranslation(0, 0.01, plt.gcf().dpi_scale_trans)
            transform = ax.transData + offset
            add_stroke_dot(ax, data[-1], max_pull, s=40, color=color, marker="^", transform=transform, path_effects=[])
    return ax

def plot_pmf(ax, input: np.array, fill_color='C0', dist_end=True, is_step=True, fill_alpha=0.5, path_effects=[pe.withStroke(linewidth=3, foreground='white')]):
    if is_step:
        x = np.arange(len(input)) + 0.5 # 生成x轴坐标点
        x = np.repeat(x, 2)             # 每个x坐标重复两次以创建阶梯效果
        y = np.repeat(input, 2)             # 每个y坐标重复两次以创建阶梯效果

        # 添加前后的额外坐标以完整显示阶梯图
        if input[0] != 0:
            x = np.append(-0.5, x[:-1])  # 在最前面加上-0.5，去掉末尾
        else:
            x = x[1:-1]
            y = y[2:]
    else:
        x = np.arange(len(input))           # 生成x轴坐标点
        y = input
        if input[0] == 0:
            x = x[1:]
            y = y[1:]
    # 绘制分布图
    ax.plot(
        x, y,
        color=fill_color,
        linewidth=1.5,
        path_effects=path_effects,
        zorder=10)
    ax.fill_between(x, 0, y, alpha=fill_alpha, color=fill_color, zorder=9)

    # 分布未结束时添加箭头
    if not dist_end:
        add_stroke_dot(ax, x[-1], y[-1], s=10, color=fill_color, marker=">", path_effects=path_effects)
        ax.plot(x, y, color=fill_color, linewidth=1.5, zorder=10)
    
    return ax

def add_vertical_quantile_pmf(ax, pdf: np.ndarray, quantile_pos:list, mark_name='抽', color='gray', pos_func=lambda x:x, pos_rate=1.1, size=10):
    '''输入pmf增加分位线'''
    cdf = np.cumsum(pdf)
    x_start = ax.get_xlim()[0]
    y_start = ax.get_ylim()[0]
    offset_v = transforms.ScaledTranslation(0, 0.05, plt.gcf().dpi_scale_trans)
    for p in quantile_pos:
        pos = np.searchsorted(cdf, p, side='left')
        if pos >= len(cdf):
            continue
        # ax.plot([x_start, pos], [cdf[pos], cdf[pos]])
        ax.plot([pos, pos], [y_start, max(pdf)*pos_rate],
                color=color, zorder=2, alpha=0.75, linestyle=':', linewidth=2)
        # 添加抽数文字
        ax.text(pos, max(pdf)*pos_rate, str(pos_func(pos))+mark_name+'\n'+str(int(p*100))+"%",
                horizontalalignment='center',
                verticalalignment='bottom',
                weight='bold',
                color='gray',
                size=size,
                transform=ax.transData + offset_v,
                path_effects=stroke_white,
                )

if __name__ == '__main__':
    import GGanalysis.games.genshin_impact as GI
    # fig, ax, _, _, y_grids, y_gap = set_square_grid_fig(max_pull=200)
    # ax.set_title('测试一下标题', weight='bold', size=18)
    # ax.set_xlabel('投入抽数', weight='medium', size=12)
    # ax.set_ylabel('获取概率', weight='medium', size=12)
    # # add_stroke_dot(ax, 0.5, 500, s=3)
    # add_vertical_cdf(
    #     ax,
    #     GI.up_5star_character(1).cdf,
    #     quantile_pos=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
    #     add_vertical_line=True,
    #     is_finite=False,
    #     add_end_mark=True,
    # )
    # description_text = get_default_description(mark_exp=GI.up_5star_character(1).exp)
    # ax.text(
    #     0, y_grids*y_gap,
    #     description_text,
    #     weight='bold',
    #     size=12,
    #     color='#B0B0B0',
    #     path_effects=stroke_white,
    #     horizontalalignment='left',
    #     verticalalignment='top'
    # )
    
    # plt.show()