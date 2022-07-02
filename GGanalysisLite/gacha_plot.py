from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties  # 字体管理器
import matplotlib.transforms as transforms
import matplotlib.patheffects as pe
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import math
from os.path import join as osp
import os
from GGanalysisLite.distribution_1d import finite_dist_1D, pad_zero

# 设置可能会使用的字体
font_w1 = font_manager.FontEntry(
    fname='C:/Windows/Fonts/SourceHanSansSC-Extralight.otf',
    name='SHS-Extralight')
font_w2 = font_manager.FontEntry(
    fname='C:/Windows/Fonts/SourceHanSansSC-Light.otf',
    name='SHS-Light')
font_w3 = font_manager.FontEntry(
    fname='C:/Windows/Fonts/SourceHanSansSC-Normal.otf',
    name='SHS-Normal')
font_w4 = font_manager.FontEntry(
    fname='C:/Windows/Fonts/SourceHanSansSC-Regular.otf',
    name='SHS-Regular')
font_w5 = font_manager.FontEntry(
    fname='C:/Windows/Fonts/SourceHanSansSC-Medium.otf',
    name='SHS-Medium')
font_w6 = font_manager.FontEntry(
    fname='C:/Windows/Fonts/SourceHanSansSC-Bold.otf',
    name='SHS-Bold')
font_w7 = font_manager.FontEntry(
    fname='C:/Windows/Fonts/SourceHanSansSC-Heavy.otf',
    name='SHS-Heavy')

font_manager.fontManager.ttflist.extend([font_w1, font_w2, font_w3, font_w4, font_w5, font_w6, font_w7])
mpl.rcParams['font.sans-serif'] = [font_w4.name]

text_font = FontProperties('SHS-Medium', size=12)
title_font = FontProperties('SHS-Bold', size=18)
mark_font = FontProperties('SHS-Bold', size=12)


class quantile_function():
    def __init__(self,
                dist_data: list=None,           # 输入数据，为包含finite_dist_1D类型的列表
                title='获取物品对抽数的分位函数', # 图表标题
                item_name='道具',               # 绘图中对道具名称的描述
                save_path='figure',             # 默认保存路径
                y_base_gap=50,                  # y轴刻度基本间隔，实际间隔为这个值的整倍数
                y2x_base=4/3,                   # 基础高宽比
                is_finite=True,                 # 是否能在有限次数内获得道具（不包括井）
                direct_exchange=None,           # 是否有井
                plot_direct_exchange=False,     # 绘图是否展示井
                max_pull=None,                  # 绘图时截断的最高抽数
                line_colors=None,               # 给出使用颜色的列表
                mark_func=None,                 # 标记道具数量的名称 如1精 6命 满潜等
                mark_offset=-0.3,               # 标记道具的标志的偏移量
                text_head=None,                 # 标记文字（前）
                text_tail=None,                 # 标记文字（后）
                ) -> None:
        # 经常修改的参数
        self.title = title
        self.item_name = item_name
        self.save_path = save_path
        for i, data in enumerate(dist_data):  # 转换numpy数组为有限一维分布类型
            if isinstance(data, np.ndarray):
                dist_data[i] = finite_dist_1D(data)
        self.data = dist_data
        self.is_finite = is_finite
        self.y_base_gap = y_base_gap
        self.y2x_base=y2x_base
        self.mark_func = mark_func
        self.mark_offset = mark_offset
        self.y_gap = self.y_base_gap
        self.x_grids = 10
        self.x_gap = 1 / self.x_grids
        self.text_head = text_head
        self.text_tail = text_tail
        self.direct_exchange = direct_exchange
        self.plot_direct_exchange = False
        if self.direct_exchange is not None:
            if plot_direct_exchange:
                self.is_finite = True
                self.plot_direct_exchange = True
        
        # 参数的默认值
        self.xlabel = '获取概率'
        self.ylabel = '投入抽数'
        self.quantile_pos = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        self.mark_pos = 0.5
        self.stroke_width = 2
        self.stroke_color = 'white'
        self.text_bias_x = -3/100 #-3/100
        self.text_bias_y = 3/100
        self.plot_path_effect = [pe.withStroke(linewidth=self.stroke_width, foreground=self.stroke_color)]

        # 处理未指定数值部分 填充默认值
        if dist_data is not None:
            self.data_num = len(self.data)
            self.exp = self.data[1].exp  # 不含井的期望
        else:
            self.data_num = 0
            self.exp = None
        if max_pull is None:
            if dist_data is None:
                self.max_pull = 100
            else:
                self.max_pull = len(self.data[-1]) - 1
        else:
            self.max_pull = max_pull
        if line_colors is None:
            self.line_colors = cm.Blues(np.linspace(0.5, 0.9, self.data_num))
        else:
            self.line_colors = line_colors

        # 计算cdf
        cdf_data = []
        for i, data in enumerate(self.data):
            if self.is_finite: 
                cdf_data.append(data.dist.cumsum())
            else:
                cdf_data.append(pad_zero(data.dist, self.max_pull)[:self.max_pull+1].cumsum())
        # 有井且需要画图的情况下的计算
        if self.plot_direct_exchange:
            calc_cdf = []
            for i in range(len(self.data)):
                if i == 0:
                    calc_cdf.append(np.ones(1, dtype=float))
                    continue
                # 遍历以前的个数 吃井次数为 i-j 次
                ans_cdf = np.copy(cdf_data[i][:i*self.direct_exchange+1])
                for j in range(1, i):
                    b_pos = self.direct_exchange*(i-j)
                    e_pos = self.direct_exchange*(i-j+1)
                    fill_ans = np.copy(cdf_data[j][b_pos:e_pos])
                    ans_cdf[b_pos:e_pos] = np.pad(fill_ans, (0, self.direct_exchange-len(fill_ans)), 'constant', constant_values=1)
                ans_cdf[i*self.direct_exchange] = 1
                calc_cdf.append(ans_cdf)
            self.cdf_data = calc_cdf
        else:
            self.cdf_data = cdf_data

    # 调试时测试
    def test_figure(self, test_pos, dpi=163):
        # 绘图部分
        self.fig, self.ax = self.set_fig()
        self.fig.set_dpi(dpi)
        # print(self.cdf_data[test_pos])
        # print(self.data[test_pos])
        for pos in test_pos:
            self.ax.plot(self.cdf_data[pos][:self.max_pull+1],
                        range(len(self.cdf_data[pos]))[:self.max_pull+1],
                        linewidth=math.log(2.5*pos), color=self.line_colors[pos])
        plt.show()
    # 绘制图像
    def show_figure(self, dpi=300, savefig=False):
        # 绘图部分
        self.fig, self.ax = self.set_fig()
        self.fig.set_dpi(dpi)
        self.add_data()
        self.add_quantile_point()
        self.add_end_mark()
        self.put_description_text()
        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            self.fig.savefig(osp(self.save_path, self.title+'.png'))
        else:
            plt.show()
    
    # 设置显示文字
    def put_description_text(self):
        description_text = ''
        # 开头附加文字
        if self.text_head is not None:
            description_text += self.text_head + '\n'
        # 对道具期望值的描述
        description_text += '获取一个'+self.item_name+'期望为'+format(self.exp, '.2f')+'抽'
        if self.direct_exchange is not None:
            description_text += '\n每'+str(self.direct_exchange)+'抽可额外兑换'+self.item_name+'\n含兑换期望为'+format(1/(1/self.exp+1/self.direct_exchange), '.2f')+'抽'
        # 对能否100%获取道具的描述
        if self.is_finite:
            max_pull = len(self.data[1])-1
            if self.direct_exchange is None:
                description_text += '\n获取一个'+self.item_name+'最多需要'+str(max_pull)+'抽'
        else:
            description_text += '\n无法确保在有限抽数内一定获得'+self.item_name
        # 末尾附加文字
        if self.text_tail is not None:
            description_text += '\n' + self.text_tail
        self.ax.text(0, self.y_grids*self.y_gap,
                        description_text,
                        fontproperties=mark_font,
                        color='#B0B0B0',
                        path_effects=self.plot_path_effect,
                        horizontalalignment='left',
                        verticalalignment='top')

    # 设置道具数量标注字符
    def get_item_num_mark(self, x):
        if self.mark_func is None:
            return str(x) + '个'
        return self.mark_func(x)

    # 在坐标轴上绘制图线
    def add_data(self):
        plot_data = self.cdf_data
        for data, color in zip(plot_data[1:], self.line_colors[1:]):
            self.ax.plot(data[:self.max_pull+1],
                        range(len(data))[:self.max_pull+1],
                        linewidth=2.5,
                        color=color)

    # 添加表示有界分布或长尾分布的标记
    def add_end_mark(self):
        # 有界分布
        if self.is_finite:
            for data, color in zip(self.cdf_data[1:], self.line_colors[1:]):
                self.ax.scatter(1, len(data)-1,
                        s=10,
                        color=color,
                        marker="o",
                        zorder=self.data_num+1,
                        path_effects=self.plot_path_effect)
        # 带井的长尾分布
        elif self.direct_exchange:
            for i, color in enumerate(self.line_colors[1:]):
                self.ax.scatter(1, (i+1)*self.direct_exchange,
                        s=10,
                        color=color,
                        marker="o",
                        zorder=self.data_num+1,
                        path_effects=self.plot_path_effect)
        # 长尾分布
        else:
            offset = transforms.ScaledTranslation(0, 0.01, self.fig.dpi_scale_trans)
            transform = self.ax.transData + offset
            self.ax.scatter(self.cdf_data[-1][-1], self.max_pull,
                        s=40,
                        color=self.line_colors[-1],
                        marker="^",
                        transform=transform,
                        zorder=self.data_num+1)

    # 在图线上绘制分位点及分位点信息
    def add_quantile_point(self):
        for i, data, color in zip(range(1, len(self.cdf_data)), self.cdf_data[1:], self.line_colors[1:]):
            for p in self.quantile_pos:
                pos = np.searchsorted(data, p, side='left')
                if pos >= len(data):
                    continue
                dot_y = (p-data[pos-1])/(data[pos]-data[pos-1])+pos-1
                offset_1 = transforms.ScaledTranslation(self.text_bias_x, self.text_bias_y, self.fig.dpi_scale_trans)
                offset_2 = transforms.ScaledTranslation(self.mark_offset+self.text_bias_x, self.text_bias_y, self.fig.dpi_scale_trans)
                offset_3 = transforms.ScaledTranslation(1.3*self.text_bias_x, self.text_bias_y, self.fig.dpi_scale_trans)
                transform_1 = self.ax.transData + offset_1
                transform_2 = self.ax.transData + offset_2
                transform_3 = self.ax.transData + offset_3
                # 这里打的点实际上不是真的点，是为了图像好看而插值到直线上的
                self.ax.scatter(p, dot_y,
                        color=color,
                        s=3,
                        zorder=self.data_num+1,
                        path_effects=self.plot_path_effect)  
                # 添加抽数文字
                self.ax.text(p, dot_y, str(pos),
                        fontproperties=text_font,
                        transform=transform_1,
                        horizontalalignment='right',
                        path_effects=self.plot_path_effect)
                # 在设置位置标注说明文字
                if p == self.mark_pos:
                    plt.text(p, dot_y, self.get_item_num_mark(i),
                        color=color,
                        fontproperties=mark_font,
                        transform=transform_2,
                        horizontalalignment='right',
                        path_effects=self.plot_path_effect)
                # 在对应最后一个道具时标记%和对应竖虚线
                if i == len(self.data)-1:
                    plt.plot([p, p], [-self.y_gap/2, dot_y+self.y_gap*2], c='gray', linewidth=2, linestyle=':')
                    plt.text(p, dot_y+self.y_gap*1.5, str(int(p*100))+"%",
                        transform=transform_3,
                        color='gray',
                        fontproperties=mark_font,
                        horizontalalignment='right',
                        path_effects=self.plot_path_effect)

    # 初始化fig，返回没有绘制线条但添加了方形网格线的fig和ax
    def set_fig(self):
        # 设置绘图大小，固定横向大小，动态纵向大小
        graph_x_space = 5       # axes横向的大小
        x_pad = 1.2             # 横向两侧留空大小
        y_pad = 1.2             # 纵向两侧留空大小
        title_y_space = 0.1     # 为标题预留大小
        x_grids = self.x_grids  # x方向网格个数
        x_gap = self.x_gap      # x方向刻度间隔
        # y刻度间隔为base_gap的整数倍 y_grids为y方向网格个数 graph_y_space为axes纵向大小
        y_gap = self.y_base_gap * math.ceil((self.max_pull / ((x_grids+1) * max(self.y2x_base, math.log(self.max_pull)/5) - 1) / self.y_base_gap))
        y_grids = math.ceil(self.max_pull / y_gap)
        graph_y_space = (y_grids + 1) * graph_x_space / (x_grids + 1)
        self.y_gap = y_gap
        self.y_grids = y_grids

        # 确定figure大小
        x_size = graph_x_space+x_pad
        y_size = graph_y_space+title_y_space+y_pad
        fig_size = [x_size, y_size]
        fig = plt.figure(figsize=fig_size) # 显示器dpi=163
        
        # 创建坐标轴
        ax = fig.add_axes([0.7*x_pad/x_size, 0.6*y_pad/y_size, graph_x_space/x_size, graph_y_space/y_size])
        # 设置标题和横纵轴标志
        ax.set_title(self.title, font=title_font)
        ax.set_xlabel(self.xlabel, font=text_font)
        ax.set_ylabel(self.ylabel, font=text_font)
        # 设置坐标轴格式，横轴为百分比显示
        ax.set_xticks(np.arange(0, 1.01, x_gap))
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.set_yticks(np.arange(0, (y_grids+2)*y_gap, y_gap))
        # 设置x，y范围，更美观
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.4*y_gap, (y_grids+0.4)*y_gap)
        # 开启主次网格和显示
        ax.grid(visible=True, which='major', color='lightgray', linestyle='-', linewidth=1)
        ax.grid(visible=True, which='minor', color='lightgray', linestyle='-', linewidth=0.5)
        ax.minorticks_on()

        return fig, ax


# text_font = FontProperties('SHS-Medium', size=12)
# title_font = FontProperties('SHS-Bold', size=18)
# mark_font = FontProperties('SHS-Bold', size=12)



# 给图增加分布列的绘制，当分布较为稀疏时采用柱状图绘制，较为密集时近似为连续函数绘制
def add_dist(   ax,
                dist:finite_dist_1D,
                is_finite=True,
                switch_len=100,
                title='所需抽数分布',
                color='C1',
                free_space = 1/40,
                show_xlabel=False
            ):
    max_pos = dist.dist.argmax()
    max_mass = dist.dist[max_pos]
    # 设置x刻度
    if len(dist) <= 100:
        if len(dist) <= 15:
            ax.set_xticks(range(0, len(dist)))
        elif len(dist) >=50:
            ax.set_xticks(np.array([0, *(range(0, len(dist), 10))[1:], len(dist)-1]))
        else:
            ax.set_xticks(np.array([0, *(range(0, len(dist), 5))[1:], len(dist)-1]))
    else:
        ax.set_xticks(np.array([0, *(range(0, len(dist), int(len(dist)/50)*5))[1:], len(dist)-1]))
    # 较稀疏时采用柱状图绘制
    if(len(dist) <= switch_len):
        if len(dist) <= 50:
            edge_width = 1.5
        else:
            edge_width = 1
        ax.bar( range(1, len(dist)), dist.dist[1:],
                color=color,
                edgecolor='black',
                linewidth=edge_width,
                zorder=10)
    # 较密集时近似为连续绘制
    else:
        ax.plot(range(1, len(dist)), dist.dist[1:],
                color=color,
                linewidth=1.5,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')],
                zorder=10)
        ax.fill_between(range(1, len(dist)), 0, dist.dist[1:], alpha=0.5, color=color, zorder=9)
        # 标记期望值与方差
        exp_y = (dist.exp-int(dist.exp)) * dist.dist[int(dist.exp)] + (int(dist.exp+1)-dist.exp) * dist.dist[int(dist.exp+1)]
        ax.axvline(x=dist.exp, c="lightgray", ls="--", lw=2, zorder=5, 
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax.text(    dist.exp+len(dist)/80, exp_y+max_mass*0.05, '期望 '+str(round(dist.exp, 2))+'抽',
                    color='gray',
                    fontproperties=mark_font,
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    zorder=11,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.scatter( dist.exp, exp_y, color='white', s=2, zorder=12,
                    path_effects=[  pe.withStroke(linewidth=5, foreground="white"),
                                    pe.withStroke(linewidth=4, foreground=color)])
        # 绘制峰值
        ax.text(    max_pos, max_mass*1.025, '峰值 '+str(max_pos)+'抽',
                    color='gray',
                    fontproperties=mark_font,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    zorder=11,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.scatter( max_pos, max_mass, color='white', s=2, zorder=11,
                    path_effects=[  pe.withStroke(linewidth=5, foreground="white"),
                                    pe.withStroke(linewidth=4, foreground=color)])
        # 标注末尾是否为长尾分布
        if is_finite:
            text_end = '最高'+str(len(dist)-1)+'抽'
        else:
            text_end = '长尾→'
        ax.text(    len(dist)-1+len(dist)/80, dist.dist[-1]+max_mass*0.05, text_end,
                    color='gray',
                    fontproperties=mark_font,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    zorder=11,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    # 绘制网格
    ax.grid(visible=True, which='major', color='lightgray', linestyle='-', linewidth=1)
    # 设置范围
    ax.set_xlim(-1-len(dist)*free_space, len(dist)+len(dist)*free_space)
    ax.set_ylim(0, max_mass*1.15)
    # 设置标题和标签
    ax.set_title(title, font=mark_font)
    if show_xlabel:
        ax.set_xlabel('抽数', fontproperties=text_font, color='gray')
    ax.set_ylabel('本抽概率', fontproperties=text_font, color='gray')
    return  ax

# 给图增加累积分布函数
def add_cdf(    ax,
                dist:finite_dist_1D,
                is_finite=True,
                title='累积分布函数',
                color='C0',
                free_space = 1/40,
                show_xlabel=True
            ):
    cdf = dist.dist.cumsum()
    # 设置x刻度
    if len(dist) <= 100:
        if len(dist) <= 15:
            ax.set_xticks(range(0, len(dist)))
        elif len(dist) >=50:
            ax.set_xticks(np.array([0, *(range(0, len(dist), 10))[1:], len(dist)-1]))
        else:
            ax.set_xticks(np.array([0, *(range(0, len(dist), 5))[1:], len(dist)-1]))
    else:
        ax.set_xticks(np.array([0, *(range(0, len(dist), int(len(dist)/50)*5))[1:], len(dist)-1]))
    # 绘制图线
    ax.plot(range(len(dist)),cdf,
            color=color,
            linewidth=3,
            path_effects=[pe.withStroke(linewidth=3, foreground='white')],
            zorder=10)
    # 绘制网格
    ax.grid(visible=True, which='major', color='lightgray', linestyle='-', linewidth=1)
    # 设置范围
    ax.set_xlim(-1-len(dist)*free_space, len(dist)+len(dist)*free_space)
    ax.set_ylim(-0.05,1.05)
    # 设置标题和标签
    ax.set_title(title, font=mark_font)
    if show_xlabel:
        ax.set_xlabel('抽数', fontproperties=text_font, color='gray')
    ax.set_ylabel('累进概率', fontproperties=text_font, color='gray')
    return ax

class draw_distribution():
    def __init__(   self,
                    dist_data=None,         # 输入数据，为finite_dist_1D类型的分布列
                    title='获取物品所需抽数分布及累进概率',
                    dpi=300,
                    figsize=(9, 8),
                    is_finite=True,         # 是否为有限分布
                ) -> None:
        # 初始化参数
        if isinstance(dist_data, np.ndarray):
            dist_data = finite_dist_1D(dist_data)
        self.data = dist_data
        self.cdf_data = self.data.dist.cumsum()
        self.fig, self.axs = plt.subplots(2, 1, constrained_layout=True)
        self.title = title
        self.dpi = dpi
        self.figsize = figsize
        self.ax_dist = self.axs[0]
        self.ax_cdf = self.axs[1]
        self.quantile_pos = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        self.is_finite = is_finite


        # 参数的默认值
        # self.mark_pos = 0.5
        # self.stroke_width = 2
        # self.stroke_color = 'white'
        # self.text_bias_x = -3/100 #-3/100
        # self.text_bias_y = 3/100
        # self.plot_path_effect = [pe.withStroke(linewidth=self.stroke_width, foreground=self.stroke_color)]

    # 设置图像参数
    def set_fig_param(self):
        self.fig.suptitle(self.title, font=title_font, size=20)
        self.fig.set_size_inches(self.figsize)
        self.fig.set_dpi(self.dpi)
 
    def test_paint(self):
        self.ax_dist = add_dist(self.ax_dist, self.data, is_finite=self.is_finite, switch_len=100)
        self.ax_cdf = add_cdf(self.ax_cdf, self.data, is_finite=self.is_finite)
        self.ax_dist.text(0, self.y_grids*self.y_gap,
                        '!!!!!!!!',
                        fontproperties=mark_font,
                        color='#B0B0B0',
                        path_effects=pe.withStroke(linewidth=2, foreground='white'),
                        horizontalalignment='left',
                        verticalalignment='top')
        plt.show()

        

# 绘制分布 需要改进
def plot_distribution(D, suptitle=None):
    D = D.dist
    # 输入的D为一维数组，从0开始
    end_pull = len(D)

    # 导入字体
    suptitle_font = FontProperties('SHS-Bold', size=15)
    
    # 自适应偏移量
    x_bias = end_pull / 100
    y_bias = 1/ 100

    # 绘图大小
    plt.figure(figsize=(9, 8))

    # 设置大图标题
    if suptitle != None:
        plt.suptitle(suptitle, fontproperties=suptitle_font)

    # 上图为累积分布函数
    plt.subplot(211)
    DD = D.cumsum()
    plt.plot(range(end_pull), DD)
    plt.title('累积分布函数', fontproperties=title_font)
    plt.xlabel('抽数', fontproperties=text_font)
    plt.ylabel('累积概率', fontproperties=text_font)
    attention_pos = [0.1, 0.25, 0.5, 0.75, 0.9, 1]
    # 设置标记
    for p in attention_pos:
        pos = np.searchsorted(DD, p, side='left')
        if pos >= len(DD):
            print(pos, p)
            continue
        # 打点标记
        plt.scatter(pos, DD[pos], s=5, zorder=10, color='slateblue', 
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])  
        # 标记%和对应竖虚线
        plt.axvline(x=pos, c="lightgray", ls="--", lw=1, zorder=0)
        plt.text(pos+2.5*x_bias, p-5*y_bias, str(pos)+'抽  '+str(int(p*100))+"%",
                c='gray',
                fontproperties=mark_font,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # 下图为分布列
    plt.subplot(212)
    plt.plot(range(end_pull), D, c='salmon')
    plt.fill_between(range(end_pull), D, np.zeros(end_pull), alpha=0.2, color='salmon')
    plt.title('分布列', fontproperties=title_font)
    plt.xlabel('抽数', fontproperties=text_font)
    plt.ylabel('本抽概率', fontproperties=text_font)

    x0 = np.arange(end_pull)
    expectation = (x0*D).sum()
    # 标记期望和对应竖虚线
    plt.axvline(x=expectation, c="lightgray", ls="--", lw=1, zorder=0)
    plt.text(expectation+1*x_bias, D.max()/2, '期望:'+str(round(expectation, 2))+'抽\n',
            c='gray',
            fontproperties=mark_font,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    # 标记分布峰值和对应竖虚线
    max_pos = D.argmax()
    y_pos_const = 0.75
    if (max_pos-expectation)/end_pull > 0.01:
        plt.axvline(x=max_pos, c="lightgray", ls="--", lw=1, zorder=0)
        y_pos_const = 0.5
    plt.text(max_pos+1*x_bias, y_pos_const*D.max(), '峰值:'+str(max_pos)+'抽',
            c='gray',
            fontproperties=mark_font,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    plt.tight_layout()
    plt.show()