from GGanalysis.ScoredItem.scored_item import *
from GGanalysis.plot_tools import *
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import os

'''
    为词条评分类道具开发的画图工具
'''

__all__ = [
    'ScoredItemDistribution',
]

# 默认显示文字函数
def get_default_score_description(
        item_name='道具',
        score_name='词条数',
        score_per_tick=None,
        text_head=None,
        mark_exp=None,
        text_tail=None
    ):
    description_text = ''
    # 开头附加文字
    if text_head is not None:
        description_text += text_head
    # 对词条数定义说明
    if score_per_tick is not None:
        if description_text != '':
            description_text += '\n'
        description_text += f'以{score_per_tick:.1f}次最小档位变化为1{score_name}'
    # 对道具得分最高值期望值的描述
    if mark_exp is not None:
        if description_text != '':
            description_text += '\n'
        description_text += '获取'+item_name+score_name+'最高值的期望为'+format(mark_exp, '.2f')
    # 末尾附加文字
    if text_tail is not None:
        if description_text != '':
            description_text += '\n'
        description_text += text_tail
    description_text =  description_text.rstrip()
    return description_text

class ScoredItemDistribution():
    def __init__(
            self,
            item: ScoredItem,
            title='道具词条数分布',
            item_name='道具',
            score_name='词条',
            save_path='figure',
            stats_name: dict={},  # 词条名称
            stats_colors: dict={},  # 词条显示颜色
            score_per_tick=10,  # 以多少内部计算分值为一显示单位
            component_use_weight=True,  # 绘制比例时按照权重绘制比例还是按照数值绘制比例
            quantile_pos=[0.01, 0.1, 0.5, 0.9, 0.99],
            show_exp=True,
            show_peak=True,
            text_head=None,
            text_tail=None,
            description_pos=0,  # 起始位置为多少评分值
            description_func=get_default_score_description,
            x_minor_ticks=10,  # 小刻度密度
        ) -> None:
        # 初始化参数
        self.title = title
        self.item = item
        self.max_score = len(self.item.score_dist)
        self.item_name = item_name
        self.score_name = score_name
        self.save_path = save_path
        self.stats_name = stats_name
        self.stats_score = item.stats_score
        self.stats_colors = stats_colors
        self.component_use_weight = component_use_weight
        self.score_per_tick = score_per_tick
        self.quantile_pos = quantile_pos
        self.text_head = text_head
        self.text_tail = text_tail
        self.description_pos = description_pos * score_per_tick
        self.description_func = description_func
        self.show_exp = show_exp
        self.show_peak = show_peak
        self.x_minor_ticks = x_minor_ticks

        # 分布最值
        self.max_pos = self.item.score_dist[:].argmax()
        self.max_mass = self.item.score_dist[self.max_pos]

        # 切换分布画图方式阈值，低于为阶梯状分布，高于为近似连续分布
        self.switch_len = 200
        
        # 这里必须重新计算一次分数，因为非整数情况分数分配到两侧导致直接套用分数导致不能归一化
        weights = np.zeros(len(item.score_dist))    # 用于归一化的分数
        self.fig_stat_ratio = {}  # 副词条占比

        # 绘制占比是按照词条数还是加权分数
        if component_use_weight:
            for key in self.item.sub_stats_exp.keys():
                if self.stats_score.get(key, np.zeros(1)) > 0:
                    weights += item.sub_stats_exp[key] * self.stats_score.get(key, np.zeros(1))
            for key in self.item.sub_stats_exp.keys():
                if self.stats_score.get(key, np.zeros(1)) > 0:
                    a = self.item.sub_stats_exp.get(key, np.zeros(1)) * self.stats_score.get(key, np.zeros(1))
                    self.fig_stat_ratio[key] = np.divide(a, weights, out=np.zeros_like(a), where=weights!=0)
        else:
            for key in self.item.sub_stats_exp.keys():
                if self.stats_score.get(key, np.zeros(1)) > 0:
                    weights += item.sub_stats_exp[key]
            for key in self.item.sub_stats_exp.keys():
                if self.stats_score.get(key, np.zeros(1)) > 0:
                    a = self.item.sub_stats_exp.get(key, np.zeros(1))
                    self.fig_stat_ratio[key] = np.divide(a, weights, out=np.zeros_like(a), where=weights!=0)

        # 描边默认值
        self.plot_path_effect = stroke_white

    def show_dist(self, figsize=(12, 5), dpi=300, savefig=False, title=None):
        '''绘制pmf'''
        # 创建 figure
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        fig.set_size_inches(figsize)
        # 设置标题
        if title is None:
            ax.set_title(f"{self.item_name}{self.score_name}分布", weight='bold', size=15)
            title = self.title
            fig.suptitle(self.title, weight='bold', size=20)
        else:
            ax.set_title(title, weight='bold', size=15)
        # 绘制分布
        self.add_score_dist(ax, quantile_pos=self.quantile_pos)
        # 保存图像
        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            fig.savefig(os.path.join(self.save_path, title+'.png'), dpi=dpi)
        else:
            plt.show()

    def show_dist_component(
            self,
            savefig=False,
            figsize=(12, 8),
            dpi=300,
            ):
        '''绘制分布及累积分布图'''
        # 两张图的创建
        fig, axs = plt.subplots(2, 1, constrained_layout=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.set_size_inches(figsize)

        ax_dist = axs[0]
        ax_component = axs[1]

        # 设置标题
        ax_dist.set_title(f"{self.item_name}{self.score_name}分布", weight='bold', size=15)
        ax_component.set_title(f"{self.score_name}{'加权' if self.component_use_weight else '未加权'}构成", weight='bold', size=15)
        fig.suptitle(self.title, weight='bold', size=20)
        # 两张图的绘制
        self.add_score_dist(ax_dist, quantile_pos=self.quantile_pos, show_xlabel=False)  # 绘制词条分布
        self.add_stats_component(ax_component)  # 绘制词条组成

        # 保存图像
        if savefig:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            fig.savefig(os.path.join(self.save_path, self.title+'.png'), dpi=dpi)
        else:
            plt.show()

    def set_xticks(self, ax):
        '''设置x刻度及xlim'''
        self.xlim_L = min(-self.score_per_tick * 0.25, -self.max_score * 0.02)
        self.xlim_R = max(self.score_per_tick * (len(self.item)/self.score_per_tick+0.25), self.max_score * 1.02)
        ax.set_xlim(self.xlim_L, self.xlim_R)
        ax.set_xticks([self.score_per_tick*i for i in range(0, int(len(self.item)/self.score_per_tick)+1)], \
                      [i for i in range(0, int(len(self.item)/self.score_per_tick)+1)])

    def add_score_dist(
                self,
                ax,
                quantile_pos=None,
                show_xlabel=True,
                show_grid=True,
                show_description=True,
                fill_alpha=0.5,
                main_color='C0',
            ):
        '''给图增加道具分数分布列的绘制，当分布较为稀疏时采用柱状图绘制，较为密集时近似为连续函数绘制（或者叫 pmf 概率质量函数）'''
        dist = self.item.score_dist

        # 设置x/y范围并设置x轴分数
        self.set_xticks(ax)
        ax.set_ylim(0, self.max_mass*1.12)
        # 设置标签
        if show_xlabel:
            ax.set_xlabel(f'{self.score_name}', weight='bold', size=12, color='black')
        ax.set_ylabel('分数分布', weight='bold', size=12, color='black')
        # 设置百分位y轴
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # 开启主次网格和显示
        if show_grid:
            ax.grid(visible=True, which='major', linestyle='-', linewidth=1)
            ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(self.x_minor_ticks))

        # 根据疏密程度切换绘图方式
        if(len(dist) <= self.switch_len):
            # 分布较短
            plot_pmf(ax, dist[:], main_color, is_step=True, fill_alpha=fill_alpha)
        else:
            # 分布较长
            plot_pmf(ax, dist[:], main_color, is_step=False, fill_alpha=fill_alpha)
        # 标记期望值与方差
        exp_y = (int(dist.exp)+1-dist.exp) * dist.dist[int(dist.exp)] + (dist.exp-int(dist.exp)) * dist.dist[int(dist.exp+1)]
        
        # 绘制分位线
        if quantile_pos is not None:
            add_vertical_quantile_pmf(ax, dist[:], mark_name=self.score_name, pos_func=lambda x:x/self.score_per_tick, quantile_pos=self.quantile_pos, pos_rate=1.05, show_mark_name=False)

        if self.show_exp:
            # 绘制期望
            ax.text(
                dist.exp+len(dist)/200, exp_y+self.max_mass*0.01, f'期望{dist.exp/self.score_per_tick:.2f}'+self.score_name,
                color='gray',
                weight='bold',
                size=10,
                horizontalalignment='center',
                verticalalignment='bottom',
                zorder=11,
                path_effects=self.plot_path_effect
            )
            # 曲线上期望处打点
            add_stroke_dot(ax, dist.exp, exp_y, color=main_color, s=10, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
        if self.show_peak:
            # 绘制峰值
            ax.text(
                self.max_pos, self.max_mass*1.01, f'峰值{self.max_pos/self.score_per_tick:.2f}'+self.score_name,
                color='gray',
                weight='bold',
                size=10,
                horizontalalignment='center',
                verticalalignment='bottom',
                zorder=11,
                path_effects=self.plot_path_effect
            )
            # 峰值处打点
            add_stroke_dot(ax, self.max_pos, self.max_mass, color=main_color, s=10, path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
        # 设置说明文字
        description_text = self.description_func(
            item_name=self.item_name,
            score_per_tick=self.score_per_tick,
            score_name=self.score_name,
            text_head=self.text_head,
            mark_exp=dist.exp/self.score_per_tick,
            text_tail=self.text_tail
            )
        if show_description:
            ax.text(
                self.description_pos, self.max_mass*0.99,
                description_text,
                weight='bold',
                size=12,
                color='#B0B0B0',
                path_effects=self.plot_path_effect,
                horizontalalignment='left',
                verticalalignment='top',
                zorder=11)
            
    def add_stats_component(
            self,
            ax,
            alpha=0.7,
            show_grid=True,
            show_xlabel=True,
        ):
        # 设置x/y范围并设置x轴分数
        self.set_xticks(ax)
        ax.set_ylim(0, 1)
        # 设置百分位y轴
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        if self.component_use_weight:
            ax.set_ylabel(f'{self.score_name}构成占比', weight='bold', size=12, color='black')
        else:
            ax.set_ylabel('未加权构成占比', weight='bold', size=12, color='black')
        # 开启主次网格和显示
        if show_grid:
            ax.grid(visible=True, which='major', linestyle='-', linewidth=1)
            ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(self.x_minor_ticks))
        # 是否绘制x轴标签
        if show_xlabel:
            ax.set_xlabel(f'{self.score_name}', weight='bold', size=12, color='black')
        # 获得 matplotlib 默认颜色循环，并使其循环
        def mpl_color_cycle():
            prop_cycle = plt.rcParams['axes.prop_cycle']
            mlp_colors = prop_cycle.by_key()['color']
            while True:
                for color in mlp_colors:
                    yield color
        color_cycler = mpl_color_cycle()  # 创建一个迭代器
        # 绘制主要累积值
        sorted_keys = sorted(self.fig_stat_ratio.keys())
        ax.stackplot(
            range(len(self.item.score_dist)), [self.fig_stat_ratio[key] for key in sorted_keys],
            labels=[self.stats_name.get(key, str(key))+' 权重:'+str(self.stats_score[key]) for key in sorted_keys],
            alpha=alpha,
            where=self.item.score_dist.dist>0,
            colors=[self.stats_colors.get(key, next(color_cycler)) for key in sorted_keys], 
            zorder=10
        )
        # 设置图例
        ax.legend(
            bbox_to_anchor=(0., -0.182*2, 1., .102),
            loc='lower left',
            fancybox=False,
            frameon=False,
            # shadow=True, 
            ncols=len(self.fig_stat_ratio.keys()),
            mode="expand",
            borderaxespad=0.,
            prop={'weight': 'normal', 'size': 10}
        )


if __name__ == '__main__':
    pass