from GGanalysis.plot_tools import *
from GGanalysis.distribution_1d import FiniteDist, pad_zero
from matplotlib.ticker import AutoMinorLocator
import matplotlib.cm as cm
import os

__all__ = [
    'QuantileFunction',
    'DrawDistribution',
]

class QuantileFunction(object):
    def __init__(self,
                dist_data: list=None,           # 输入数据，为包含finite_dist_1D类型的列表
                title='获取物品对抽数的分位函数', # 图表标题
                item_name='道具',               # 绘图中对道具名称的描述
                save_path='figure',             # 默认保存路径
                y_base_gap=50,                  # y轴刻度基本间隔，实际间隔为这个值的整倍数
                y2x_base=4/3,                   # 基础高宽比
                y_force_gap=None,               # y轴强制间隔
                is_finite=True,                 # 是否能在有限次数内达到目标（不包括井）
                direct_exchange=None,           # 是否有井
                plot_direct_exchange=False,     # 绘图是否展示井
                max_pull=None,                  # 绘图时截断的最高抽数
                line_colors=None,               # 给出使用颜色的列表
                mark_func=default_item_num_mark,# 标记道具数量的名称 如1精 6命 满潜等
                mark_offset=-0.3,               # 标记道具的标志的偏移量
                text_head=None,                 # 标记文字（前）
                text_tail=None,                 # 标记文字（后）
                mark_exp=True,                  # 是否在图中标注期望值
                mark_max_pull=True,             # 是否在图中标注最多需要抽数
                description_func=get_default_description,
                cost_name='抽'
                ) -> None:
        # 输入检查
        if line_colors is not None and len(dist_data) != len(line_colors):
            raise ValueError("Item number must match colors!")
        
        # 经常修改的参数
        self.title = title
        self.item_name = item_name
        self.save_path = save_path
        for i, data in enumerate(dist_data):  # 转换numpy数组为有限一维分布类型
            if isinstance(data, np.ndarray):
                dist_data[i] = FiniteDist(data)
        self.data = dist_data
        self.is_finite = is_finite
        self.y_base_gap = y_base_gap
        self.y2x_base = y2x_base
        self.y_force_gap = y_force_gap
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
        
        self.description_func = description_func
        self.cost_name = cost_name
        
        # 参数的默认值
        self.xlabel = '获取概率'
        self.ylabel = f'投入{self.cost_name}数'
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
                cdf_data.append(data.cdf)
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

    # 绘制图像
    def show_figure(self, dpi=300, savefig=False):
        # 绘图部分
        fig, ax, _, _, y_grids, y_gap = set_square_grid_fig(
            max_pull=self.max_pull,
            x_grids=self.x_grids,
            y_base_gap=self.y_base_gap,
            y2x_base=self.y2x_base,
            y_force_gap=self.y_force_gap
        )
        fig.set_dpi(dpi)
        ax.set_title(self.title, weight='bold', size=18)
        ax.set_xlabel('获取概率', weight='medium', size=12)
        ax.set_ylabel(f'投入{self.cost_name}数', weight='medium', size=12)
        
        # 分道具数量添加分位函数
        for i, (data, color) in enumerate(zip(self.cdf_data[1:], self.line_colors[1:])):
            add_quantile_line(
                ax,
                data[:self.max_pull+1],
                color=color,
                item_num=i+1,
                linewidth=2.5,
                quantile_pos=self.quantile_pos,
                add_vertical_line=(i+1==len(self.cdf_data)-1),
                is_finite=self.is_finite,
                add_end_mark=True,
                mark_func=self.mark_func,
            )

        # 根据传入函数生成描述文本
        description_text = self.description_func(
            item_name=self.item_name,
            cost_name=self.cost_name,
            text_head=self.text_head,
            mark_exp=self.exp,
            direct_exchange=self.direct_exchange,
            show_max_pull=len(self.data[1])-1,  # 对于非有限情况也可以传进去，会自动判断
            is_finite=self.is_finite,
            text_tail=self.text_tail,
            )  # 不含井的情况

        # 添加上描述文本
        ax.text(
            0, y_grids*y_gap,
            description_text,
            weight='bold',
            size=12,
            color='#B0B0B0',
            path_effects=stroke_white,
            horizontalalignment='left',
            verticalalignment='top'
        )

        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            fig.savefig(os.path.join(self.save_path, self.title+'.png'), dpi=dpi)
        else:
            plt.show()

class DrawDistribution(object):
    def __init__(   self,
                    dist_data=None,         # 输入数据，为finite_dist_1D类型的分布列
                    max_pull=None,
                    title='获取物品所需抽数分布及累进概率',
                    item_name='道具',
                    cost_name='抽',
                    save_path='figure',
                    show_description=True,
                    quantile_pos=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
                    description_pos=0,
                    show_exp=True,
                    show_peak=True,
                    is_finite=True,
                    text_head=None,
                    text_tail=None,
                    description_func=get_default_description,
                ) -> None:
        # 初始化参数
        if isinstance(dist_data, np.ndarray):
            dist_data = FiniteDist(dist_data)
        self.dist_data = dist_data
        self.max_pull = len(dist_data) if max_pull is None else max_pull
        self.show_description = show_description
        self.title = title
        self.item_name = item_name
        self.cost_name = cost_name
        self.text_head = text_head
        self.text_tail = text_tail
        self.description_pos = description_pos
        self.save_path = save_path
        self.description_func = description_func
        self.show_exp = show_exp
        self.show_peak = show_peak

        # 绘图分位点
        self.quantile_pos = quantile_pos
        self.is_finite = is_finite

        # 绘图时横向两侧留空
        self.x_free_space = 1/40
        self.x_left_lim = 0-self.max_pull*self.x_free_space
        self.x_right_lim = self.max_pull+self.max_pull*self.x_free_space
        # 分布最值
        self.max_pos = dist_data[:].argmax()
        self.max_mass = dist_data.dist[self.max_pos]

        # 切换分布画图方式阈值，低于为阶梯状分布，高于为近似连续分布
        self.switch_len = 200
        # 描边默认值
        self.plot_path_effect = stroke_white

    def show_dist(self, figsize=(9, 5), dpi=300, savefig=False, title=None):
        '''绘制pmf'''
        # 创建 figure
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(figsize)
        # 设置标题
        if title is None:
            ax.set_title(f"所需{self.cost_name}数分布", weight='bold', size=15)
            title = self.title
            fig.suptitle(self.title, weight='bold', size=20)
        else:
            ax.set_title(title, weight='bold', size=15)
        # 绘制分布
        self.add_dist(ax, quantile_pos=self.quantile_pos)
        # 保存图像
        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            fig.savefig(os.path.join(self.save_path, title+'.png'), dpi=dpi)
        else:
            plt.show()

    def show_cdf(self, figsize=(9, 5), dpi=300, savefig=False, title=None):
        '''绘制cdf'''
        # 一张图的创建
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(figsize)
        # 设置标题
        if title is None:
            ax.set_title("累积分布函数", weight='bold', size=15)
            title = self.title
            fig.suptitle(self.title, weight='bold', size=20)
        else:
            ax.set_title(title, weight='bold', size=15)
        # 绘制累积概率函数
        self.add_cdf(ax, show_title=False, show_xlabel=True)
        # 保存图像
        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            fig.savefig(os.path.join(self.save_path, title+'.png'), dpi=dpi)
        else:
            plt.show()   

    def show_two_graph(
            self,
            savefig=False,
            figsize=(9, 8),
            dpi=300,
            color='C0',
            ):
        '''绘制分布及累积分布图'''
        # 两张图的创建
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.set_size_inches(figsize)
        '''
        # 创建一个新的 GridSpec 实例
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 3], figure=fig)
        # 重新设置 axs 中每个子图的位置和大小
        axs[0].set_position(gs[0].get_position(fig))
        axs[1].set_position(gs[1].get_position(fig))
        '''
        ax_dist = axs[0]
        ax_cdf = axs[1]

        # 设置标题
        fig.suptitle(self.title, weight='bold', size=20)
        ax_dist.set_title(f"所需{self.cost_name}数分布", weight='bold', size=15)
        ax_cdf.set_title("累积分布函数", weight='bold', size=15)
        # 两张图的绘制
        # 绘制分布
        self.add_dist(ax_dist, quantile_pos=self.quantile_pos, show_xlabel=False)
        ax_dist.set_title(f"所需{self.cost_name}数分布", weight='bold', size=15)
        
        # 绘制累积概率函数
        self.add_cdf(ax_cdf, show_title=False, show_xlabel=True)
        # 保存图像
        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            fig.savefig(os.path.join(self.save_path, self.title+'.png'), dpi=dpi)
        else:
            plt.show()

    def add_dist(  
                    self, 
                    ax,
                    quantile_pos=None,
                    show_xlabel=True,
                    show_grid=True,
                    show_description=True,
                    fill_alpha=0.5,
                    minor_ticks=10,
                    color='C0',
                ):
        '''给图增加分布列的绘制，当分布较为稀疏时采用柱状图绘制，较为密集时近似为连续函数绘制（或者叫 pmf 概率质量函数）'''
        dist = self.dist_data
        # 设置x/y范围
        ax.set_xlim(self.x_left_lim, self.x_right_lim)
        ax.set_ylim(0, self.max_mass*1.26)
        # 设置标签
        if show_xlabel:
            ax.set_xlabel(f'{self.cost_name}数', weight='bold', size=12, color='black')
        ax.set_ylabel(f'本{self.cost_name}概率', weight='bold', size=12, color='black')
        # 开启主次网格和显示
        if show_grid:
            ax.grid(visible=True, which='major', linestyle='-', linewidth=1)
            ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks))
        
        # 根据疏密程度切换绘图方式
        if(len(dist) <= self.switch_len):
            # 分布较短
            plot_pmf(ax, dist[:self.max_pull], color, self.is_finite, is_step=True, fill_alpha=fill_alpha)
        else:
            # 分布较长
            plot_pmf(ax, dist[:self.max_pull], color, self.is_finite, is_step=False, fill_alpha=fill_alpha)
        # 标记期望值与方差
        exp_y = (int(dist.exp)+1-dist.exp) * dist.dist[int(dist.exp)] + (dist.exp-int(dist.exp)) * dist.dist[int(dist.exp+1)]
        # ax.axvline(x=dist.exp, c="lightgray", ls="--", lw=2, zorder=5, 
        #             path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        
        # 绘制分位线
        if quantile_pos is not None:
            add_vertical_quantile_pmf(ax, dist[:self.max_pull], quantile_pos=self.quantile_pos)

        if self.show_exp:
            # 绘制期望
            ax.text(
                dist.exp+len(dist)/200, exp_y+self.max_mass*0.01, '期望'+str(round(dist.exp, 1))+self.cost_name,
                color='gray',
                weight='bold',
                size=10,
                horizontalalignment='center',
                verticalalignment='bottom',
                zorder=11,
                path_effects=self.plot_path_effect
            )
            # 曲线上期望处打点
            add_stroke_dot(ax, dist.exp, exp_y, color=color, s=10, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
        if self.show_peak:
            # 绘制峰值
            ax.text(
                self.max_pos, self.max_mass*1.01, '峰值'+str(self.max_pos)+self.cost_name,
                color='gray',
                weight='bold',
                size=10,
                horizontalalignment='center',
                verticalalignment='bottom',
                zorder=11,
                path_effects=self.plot_path_effect
            )
            # 峰值处打点
            add_stroke_dot(ax, self.max_pos, self.max_mass, color=color, s=10, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

        # 设置说明文字
        description_text = self.description_func(
            item_name=self.item_name,
            cost_name=self.cost_name,
            text_head=self.text_head,
            mark_exp=self.dist_data.exp,
            show_max_pull=len(dist)-1,  # 对于非有限情况也可以传进去，会自动判断
            is_finite=self.is_finite,
            text_tail=self.text_tail,
            )  # 不含井的情况
        if show_description:
            ax.text(
                self.description_pos, self.max_mass*1.1,
                description_text,
                weight='bold',
                size=12,
                color='#B0B0B0',
                path_effects=self.plot_path_effect,
                horizontalalignment='left',
                verticalalignment='top',
                zorder=11)
    
    # 给图增加累积质量函数
    def add_cdf(    
                    self,
                    ax,
                    title='累积分布函数',
                    main_color='C0',
                    minor_ticks=10,
                    show_title=True,
                    show_grid=True,
                    show_xlabel=True,
                    show_description=False,
                ):
        dist = self.dist_data
        cdf = dist.dist.cumsum()

        # 设置标题和标签
        if show_title:
            ax.set_title(title, weight='bold', size=15)
        if show_xlabel:
            ax.set_xlabel(f"{self.cost_name}数", weight='bold', size=12, color='black')
        ax.set_ylabel('累进概率', weight='bold', size=12, color='black')
        # 设置x/y范围和刻度
        ax.set_xlim(self.x_left_lim, self.x_right_lim)
        ax.set_ylim(-0.05,1.15)
        ax.set_yticks(np.linspace(0, 1, 11))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        # 开启主次网格和显示
        if show_grid:
            ax.grid(visible=True, which='major', linestyle='-', linewidth=1)
            ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

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
        offset = transforms.ScaledTranslation(-0.05, 0.01, plt.gcf().dpi_scale_trans)
        trans = ax.transData + offset
        for p in self.quantile_pos:
            pos = np.searchsorted(cdf, p, side='left')
            if pos >= len(cdf): # 长度超界
                continue
            # ax.plot([self.x_left_lim-1, pos], [cdf[pos], cdf[pos]], c="lightgray", linewidth=2, linestyle="--", zorder=0)
            ax.plot([pos, pos], [-1, cdf[pos]], c="lightgray", linewidth=2, linestyle="--", zorder=0)
            add_stroke_dot(ax, pos, cdf[pos], color=main_color, s=10, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
            ax.text(pos, cdf[pos], str(pos)+f'{self.cost_name}\n'+str(round(p*100))+'%',
                    weight='bold',
                    size=10,
                    color='gray',
                    transform=trans,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    path_effects=self.plot_path_effect)
        
        # 设置说明文字
        description_text = self.description_func(
            item_name=self.item_name,
            cost_name=self.cost_name,
            text_head=self.text_head,
            mark_exp=self.dist_data.exp,
            show_max_pull=len(dist)-1,  # 对于非有限情况也可以传进去，会自动判断
            is_finite=self.is_finite,
            text_tail=self.text_tail,
            )  # 不含井的情况
        if show_description:
            ax.text(
                self.description_pos, 1.033,
                description_text,
                weight='bold',
                size=12,
                color='#B0B0B0',
                path_effects=self.plot_path_effect,
                horizontalalignment='left',
                verticalalignment='top',
                zorder=11)

if __name__ == '__main__':
    # 测试分位图绘制
    import GGanalysis.games.genshin_impact as GI
    import time
    # fig = QuantileFunction(
    #     GI.up_5star_character(7, multi_dist=True),
    #     title='测试标题',
    #     item_name='UP五星角色',
    #     text_head='采用官方公示模型',
    #     text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    #     max_pull=1400,
    #     # mark_func=,
    #     line_colors=cm.YlOrBr(np.linspace(0.4, 0.9, 7+1)),  # cm.OrAKges(np.linspace(0.5, 0.9, 6+1)),
    #     # y_base_gap=25,
    #     # y2x_base=2,
    #     is_finite=True)
    # fig.show_figure(dpi=300, savefig=True)
    pass
    a = GI.common_5star(1)
    fig = DrawDistribution(
        a,
        item_name='五星道具',
        quantile_pos=[0.1, 0.2, 0.3, 0.5, 0.9],
        text_head='采用官方公示模型',
        text_tail='@一棵平衡树 '+time.strftime('%Y-%m-%d',time.localtime(time.time())),
    )
    fig.show_dist()
    # fig.show_cdf()
    # fig.show_two_graph()
    