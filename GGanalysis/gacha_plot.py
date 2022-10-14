from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties  # 字体管理器
import matplotlib.transforms as transforms
import matplotlib.patheffects as pe
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import math
import os.path as osp
import os
import sys
from GGanalysis.distribution_1d import finite_dist_1D, pad_zero

font_path = None
if sys.platform == 'win32':  # windows下
    font_path = 'C:/Windows/Fonts/'
else:  # Linux
    font_path = os.path.expanduser('~/.local/share/fonts/')

# 设置可能会使用的字体
font_w1 = font_manager.FontEntry(
    fname=font_path+'SourceHanSansSC-Extralight.otf',
    name='SHS-Extralight')
font_w2 = font_manager.FontEntry(
    fname=font_path+'SourceHanSansSC-Light.otf',
    name='SHS-Light')
font_w3 = font_manager.FontEntry(
    fname=font_path+'SourceHanSansSC-Normal.otf',
    name='SHS-Normal')
font_w4 = font_manager.FontEntry(
    fname=font_path+'SourceHanSansSC-Regular.otf',
    name='SHS-Regular')
font_w5 = font_manager.FontEntry(
    fname=font_path+'SourceHanSansSC-Medium.otf',
    name='SHS-Medium')
font_w6 = font_manager.FontEntry(
    fname=font_path+'SourceHanSansSC-Bold.otf',
    name='SHS-Bold')
font_w7 = font_manager.FontEntry(
    fname=font_path+'SourceHanSansSC-Heavy.otf',
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
                mark_exp=True,                  # 是否在图中标注期望值
                mark_max_pull=True,             # 是否在图中标注最多需要抽数
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
        self.mark_exp = mark_exp
        self.mark_max_pull = mark_max_pull
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
            if not osp.exists(self.save_path):
                os.makedirs(self.save_path)
            self.fig.savefig(osp.join(self.save_path, self.title+'.png'), dpi=dpi)
        else:
            plt.show()
    
    # 设置显示文字
    def put_description_text(self):
        description_text = ''
        # 开头附加文字
        if self.text_head is not None:
            description_text += self.text_head
        # 对道具期望值的描述
        if self.mark_exp:
            description_text += '\n获取一个'+self.item_name+'期望为'+format(self.exp, '.2f')+'抽'
        if self.direct_exchange is not None:
            description_text += '\n每'+str(self.direct_exchange)+'抽可额外兑换'+self.item_name+'\n含兑换期望为'+format(1/(1/self.exp+1/self.direct_exchange), '.2f')+'抽'
        # 对能否100%获取道具的描述
        if self.mark_max_pull:
            if self.is_finite:
                max_pull = len(self.data[1])-1
                if self.direct_exchange is None:
                    description_text += '\n获取一个'+self.item_name+'最多需要'+str(max_pull)+'抽'
            else:
                description_text += '\n无法确保在有限抽数内一定获得'+self.item_name
        # 末尾附加文字
        if self.text_tail is not None:
            description_text += '\n' + self.text_tail
        description_text =  description_text.rstrip()
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
                    self.ax.plot([p, p], [-self.y_gap/2, dot_y+self.y_gap*2], c='gray', linewidth=2, linestyle=':')
                    self.ax.text(p, dot_y+self.y_gap*1.5, str(int(p*100))+"%",
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


class draw_distribution():
    def __init__(   self,
                    dist_data=None,         # 输入数据，为finite_dist_1D类型的分布列
                    current_pulls=None,
                    future_pulls=None,
                    max_pull=None,
                    title='获取物品所需抽数分布及累进概率',
                    dpi=300,
                    show_description=True,
                    is_finite=True,         # 是否为有限分布
                ) -> None:
        # 初始化参数
        if isinstance(dist_data, np.ndarray):
            dist_data = finite_dist_1D(dist_data)
        if max_pull is not None:
            dist_data.dist = dist_data.dist[:max_pull+1]
        self.data = dist_data
        self.current_pulls = current_pulls
        self.future_pulls = future_pulls
        self.show_description = show_description
        self.cdf_data = self.data.dist.cumsum()
        self.title = title
        self.dpi = dpi
        
        # 绘图分位点
        self.quantile_pos = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        self.is_finite = is_finite

        # 绘图时横向两侧留空
        self.x_free_space = 1/40
        self.x_left_lim = 0-len(self.data)*self.x_free_space
        self.x_right_lim = len(self.data)+len(self.data)*self.x_free_space
        # 分布最值
        self.max_pos = dist_data.dist.argmax()
        self.max_mass = dist_data.dist[self.max_pos]

        # 切换分布画图方式阈值，低于为柱状图，高于为折线图
        self.switch_len = 100
        # 描边默认值
        self.plot_path_effect = [pe.withStroke(linewidth=2, foreground="white")]

    # 绘制分布及累积分布图
    def draw_two_graph(
            self,
            savefig=False,
            figsize=(9, 8), 
            main_color='royalblue',
            current_color='limegreen',
            future_color='orange',
            ):
        # 两张图的创建
        self.fig, self.axs = plt.subplots(2, 1, constrained_layout=True)
        self.ax_dist = self.axs[0]
        self.ax_cdf = self.axs[1]
        self.set_fig_param(figsize=figsize)
        # 两张图的绘制
        self.set_xticks(self.ax_dist)
        self.set_xticks(self.ax_cdf)
        self.add_dist(self.ax_dist, main_color=main_color, current_color=current_color, future_color=future_color)
        self.add_cdf(self.ax_cdf)
        if savefig:
            plt.savefig(self.title+'.png', dpi=self.dpi)
        else:
            plt.show()

    def draw_dist(self, savefig=False, figsize=(9, 5)):
        # 一张图的创建
        self.fig, self.ax_dist = plt.subplots(1, 1, constrained_layout=True)
        self.set_fig_param(figsize=figsize)
        # 两张图的绘制
        self.set_xticks(self.ax_dist)
        self.add_dist(self.ax_dist, show_title=False, show_xlabel=True)
        if savefig:
            plt.savefig(self.title+'.png', dpi=self.dpi)
        else:
            plt.show()

    def draw_cdf(self, savefig=False, figsize=(9, 5)):
        # 一张图的创建
        self.fig, self.ax_cdf = plt.subplots(1, 1, constrained_layout=True)
        self.set_fig_param(figsize=figsize)
        # 两张图的绘制
        self.set_xticks(self.ax_cdf)
        self.add_cdf(self.ax_cdf, show_title=False, show_xlabel=True)
        if savefig:
            plt.savefig(self.title+'.png', dpi=self.dpi)
        else:
            plt.show()

    # 设置图像参数
    def set_fig_param(self, figsize=(9, 8)):
        self.fig.suptitle(self.title, font=title_font, size=20)
        self.fig.set_size_inches(figsize)
        self.fig.set_dpi(self.dpi)
    
    # 图像打点
    def add_point(self, ax, x, y, color):
        ax.scatter( x, y, color='white', s=2, zorder=12,
                    path_effects=[  pe.withStroke(linewidth=5, foreground="white"),
                                    pe.withStroke(linewidth=4, foreground=color)])

    def test_paint(self):
        self.set_xticks(self.ax_dist)
        self.set_xticks(self.ax_cdf)
        self.add_dist(self.ax_dist)
        self.add_cdf(self.ax_cdf)
        plt.show()
    
    # 设置x刻度及xlim
    def set_xticks(self, ax):
        dist_len = len(self.data)
        if dist_len <= self.switch_len:
            # 不同长度段有不同策略
            if dist_len <= 15:
                ax.set_xticks([1, *range(0, dist_len)[1:]])
            elif dist_len <= 50:
                ax.set_xticks(np.array([1, *(range(0, dist_len, 5))[1:], dist_len-1]))
            elif dist_len <= 100:
                ax.set_xticks(np.array([1, *(range(0, dist_len, 10))[1:], dist_len-1]))
            # 太长的情况，交给默认值
            else:
                pass
        else:
            ax.set_xticks(np.array([0, *(range(0, dist_len, int(dist_len/50)*5))[1:], dist_len-1]))
        ax.set_xlim(self.x_left_lim, self.x_right_lim)

    # 给图增加分布列的绘制，当分布较为稀疏时采用柱状图绘制，较为密集时近似为连续函数绘制
    def add_dist(  
                    self, ax,
                    title='所需抽数分布',
                    show_title=True,
                    main_color='royalblue',
                    current_color='limegreen',
                    future_color='orange',
                    show_xlabel=False
                ):
        dist = self.data
        color_exp = main_color
        color_peak = main_color
        # 较稀疏时采用柱状图绘制
        if(len(dist) <= self.switch_len):
            if len(dist) <= 50:
                edge_width = 1.5
            else:
                edge_width = 1
            ax.bar( range(1, len(dist)), dist.dist[1:],
                    color=main_color,
                    edgecolor='black',
                    linewidth=edge_width,
                    zorder=10)
            # 绘制峰值
            ax.text(    self.max_pos, self.max_mass*1.025, '峰值 '+str(self.max_pos)+'抽',
                        color='gray',
                        fontproperties=mark_font,
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        zorder=11,
                        path_effects=self.plot_path_effect)
        # 较密集时近似为连续绘制
        else:
            def draw_color_region(begin, end, fill_color):
                ax.plot(range(begin, end), dist.dist[begin: end],
                    color=fill_color,
                    linewidth=1.5,
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')],
                    zorder=10)
                ax.fill_between(range(begin, end), 0, dist.dist[begin: end], alpha=0.5, color=fill_color, zorder=9)
                
            begin_pulls = 1
            if self.current_pulls:
                draw_color_region(begin_pulls, min(len(dist)-1,self.current_pulls+1), current_color)
                # 曲线上分界处打点
                ax.axvline(x=self.current_pulls, c="lightgray", ls="--", lw=2, zorder=5, 
                            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
                ax.text(    self.current_pulls, dist[min(len(dist)-1,self.current_pulls)]+self.max_mass*0.05,
                            '当前 '+str(self.current_pulls)+'抽\n'+str(round(sum(dist[1:min(self.current_pulls, len(dist))])*100))+'%',
                            color='gray',
                            fontproperties=mark_font,
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            zorder=12,
                            path_effects=self.plot_path_effect)
                self.add_point(ax, self.current_pulls, dist[min(len(dist)-1,self.current_pulls)], current_color)
                # 设置颜色
                if self.max_pos <= self.current_pulls and self.max_pos >= begin_pulls:
                    color_peak = current_color
                if dist.exp <= self.current_pulls and dist.exp >= begin_pulls:
                    color_exp = current_color

                begin_pulls = min(len(dist), self.current_pulls)
            if self.future_pulls and begin_pulls <= self.future_pulls:
                draw_color_region(begin_pulls, min(len(dist),self.future_pulls+1), future_color)
                # 曲线上分界处打点
                ax.axvline(x=self.future_pulls, c="lightgray", ls="--", lw=2, zorder=5, 
                            path_effects=[pe.withStroke(linewidth=3, foreground="white")])
                ax.text(    self.future_pulls, dist[min(len(dist)-1, self.future_pulls)]+self.max_mass*0.05,
                            '未来 '+str(self.future_pulls)+'抽\n'+str(round(sum(dist[1:min(self.future_pulls, len(dist))])*100))+'%',
                            color='gray',
                            fontproperties=mark_font,
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            zorder=12,
                            path_effects=self.plot_path_effect)
                self.add_point(ax, self.future_pulls,  dist[min(len(dist)-1, self.future_pulls)], future_color)
                # 设置颜色
                if self.max_pos <= self.future_pulls and self.max_pos >= begin_pulls:
                    color_peak = future_color
                if dist.exp <= self.future_pulls and dist.exp >= begin_pulls:
                    color_exp = future_color

                begin_pulls = min(len(dist), self.future_pulls)
            draw_color_region(begin_pulls, len(dist), main_color)

            # 旧代码
            # ax.plot(range(1, len(dist)), dist.dist[1:],
            #         color=main_color,
            #         linewidth=1.5,
            #         path_effects=[pe.withStroke(linewidth=3, foreground='white')],
            #         zorder=10)
            # ax.fill_between(range(1, len(dist)), 0, dist.dist[1:], alpha=0.5, color=main_color, zorder=9)
            # 标记期望值与方差
            exp_y = (int(dist.exp)+1-dist.exp) * dist.dist[int(dist.exp)] + (dist.exp-int(dist.exp)) * dist.dist[int(dist.exp+1)]
            ax.axvline(x=dist.exp, c="lightgray", ls="--", lw=2, zorder=5, 
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")])
            ax.text(    dist.exp+len(dist)/80, exp_y+self.max_mass*0.05, '期望 '+str(round(dist.exp, 1))+'抽',
                        color='gray',
                        fontproperties=mark_font,
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        zorder=11,
                        path_effects=self.plot_path_effect)
            # 曲线上期望处打点
            self.add_point(ax, dist.exp, exp_y, color_exp)
            # 判断是否有足够空间绘制峰值
            if abs(self.max_pos - dist.exp) / len(dist) > 0.05:
                # 绘制峰值
                ax.text(    self.max_pos, self.max_mass*1.025, '峰值 '+str(self.max_pos)+'抽',
                            color='gray',
                            fontproperties=mark_font,
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            zorder=11,
                            path_effects=self.plot_path_effect)
                # 峰值处打点
                self.add_point(ax, self.max_pos, self.max_mass, color_peak)
            # 标注末尾是否为长尾分布
            if self.is_finite is False:
                ax.scatter( len(dist)-1, self.data.dist[len(dist)-1],
                            s=80, color=main_color, marker=">", zorder=11)
        # 绘制网格
        ax.grid(visible=True, which='major', color='lightgray', linestyle='-', linewidth=1)
        # 设置y范围
        ax.set_ylim(0, self.max_mass*1.15)
        # 设置标题和标签
        if show_title:
            ax.set_title(title, font=mark_font)
        if show_xlabel:
            ax.set_xlabel('抽数', fontproperties=text_font, color='black')
        ax.set_ylabel('本抽概率', fontproperties=text_font, color='black')
        # 设置说明文字
        show_text = '获取道具的期望为'+str(round(dist.exp, 2))+'抽\n分布峰值位于第'+str(self.max_pos)+'抽'
        if self.is_finite:
            show_text += '\n获取道具最多需要'+str(len(dist)-1)+'抽'
        else:
            show_text += '\n无法保证在有限抽内获得道具'
        if self.current_pulls:
            show_text += '\n当前手上有'+str(self.current_pulls)+'抽'
        if self.future_pulls:
            show_text += '\n预计未来有'+str(self.future_pulls)+'抽'
        if self.show_description:
            ax.text(0, self.max_mass*1.08,
                    show_text,
                    fontproperties=mark_font,
                    color='#B0B0B0',
                    path_effects=self.plot_path_effect,
                    horizontalalignment='left',
                    verticalalignment='top',
                    zorder=11)
        
    # 给图增加累积分布函数
    def add_cdf(    
                    self,
                    ax,
                    title='累积分布函数',
                    show_title=True,
                    main_color='C0',
                    show_xlabel=True
                ):
        dist = self.data
        cdf = dist.dist.cumsum()
        # 绘制图线
        ax.plot(range(1, len(dist)),cdf[1:],
                color=main_color,
                linewidth=3,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')],
                zorder=10)
        # 标注末尾是否为长尾分布
        if self.is_finite is False:
            ax.scatter( len(cdf)-1, cdf[len(dist)-1],
                        s=80, color=main_color, marker=">", zorder=11)
        # 绘制网格
        ax.grid(visible=True, which='major', color='lightgray', linestyle='-', linewidth=1)
        # 绘制分位点及标注文字
        offset = transforms.ScaledTranslation(-0.05, 0.01, self.fig.dpi_scale_trans)
        trans = ax.transData + offset
        for p in self.quantile_pos:
            pos = np.searchsorted(cdf, p, side='left')
            if pos >= len(cdf): # 长度超界
                continue
            ax.plot([self.x_left_lim-1, pos], [cdf[pos], cdf[pos]], c="lightgray", linewidth=2, linestyle="--", zorder=5, 
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
            ax.plot([pos, pos], [-1, cdf[pos]], c="lightgray", linewidth=2, linestyle="--", zorder=5, 
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
            self.add_point(ax, pos, cdf[pos], main_color)
            ax.text(pos, cdf[pos], str(pos)+'抽 '+str(round(p*100))+'%',
                    fontproperties=mark_font,
                    color='gray',
                    transform=trans,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    path_effects=self.plot_path_effect)

        # 设置范围和y刻度
        ax.set_ylim(-0.05,1.1)
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        # 设置标题和标签
        if show_title:
            ax.set_title(title, font=mark_font)
        if show_xlabel:
            ax.set_xlabel('抽数', fontproperties=text_font, color='black')
        ax.set_ylabel('累进概率', fontproperties=text_font, color='black')
        # 设置说明文字
        show_text = ''
        if self.is_finite:
            show_text += '获取道具最多需要'+str(len(dist)-1)+'抽'
        else:
            show_text += '无法保证在有限抽内获得道具'
        if self.show_description:
            ax.text(0, 1.033,
                    show_text,
                    fontproperties=mark_font,
                    color='#B0B0B0',
                    path_effects=self.plot_path_effect,
                    horizontalalignment='left',
                    verticalalignment='top',
                    zorder=11)

