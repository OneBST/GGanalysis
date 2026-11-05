'''幻乐秀场「群星乐透」星琼消耗分布图。

从仓库根目录运行：
python -m GGanalysis.games.honkai_star_rail.figure_plot_phantasia_showstage
图片保存至 figure/，同时打印八个档位的概率和期望。
'''
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

from GGanalysis.gacha_plot import DrawDistribution
from GGanalysis.games.honkai_star_rail.gacha_model import phantasia_showstage
from GGanalysis.plot_tools import stroke_white


# 展示时按跃迁单抽价格折算，与本活动实际抽取次数不同。
STELLAR_JADE_PER_WARP = 160


def plot_phantasia_showstage(save_path: str | Path = 'figure') -> Figure:
    '''复用 DrawDistribution 双图模板，保存 PNG/SVG 并返回 Figure。'''
    jade_dist = phantasia_showstage('stellar_jade')
    costs = np.flatnonzero(jade_dist.dist)
    probabilities = jade_dist.dist[costs]
    cumulative = jade_dist.cdf[costs]
    equivalent_warps = jade_dist.exp / STELLAR_JADE_PER_WARP
    output = Path(save_path)
    output.mkdir(parents=True, exist_ok=True)

    # 沿用异环示例的双图模板和项目默认样式。
    # 不使用分位点标注，改为对每个实际花费档位标注概率。
    distribution_plot = DrawDistribution(
        dist_data=jade_dist,
        title='星穹铁道「群星乐透」时装获取星琼消耗分布',
        item_name='「群星乐透」时装',
        cost_name='星琼',
        save_path=str(output),
        text_head='采用4.7活动公示数据，从未参与开始抽取\n'
                  '首抽免费，100星琼兑换1张时装邀请函',
        text_tail=f'星轨专票按{STELLAR_JADE_PER_WARP}星琼/张折算\n'
                  '@一棵平衡树 2026-10-30',
        quantile_pos=[],
        is_finite=True,
        show_peak=False,
        show_exp=False,
    )
    # 星琼数组虽长，但只有八个离散花费档位，应采用模板的阶梯绘制。
    distribution_plot.switch_len = len(jade_dist)
    # 两端各留 500 星琼空间，容纳首尾档位的柱子和文字。
    distribution_plot.x_left_lim = -500
    distribution_plot.x_right_lim = int(costs[-1]) + 500
    distribution_plot.show_two_graph(savefig=True, figsize=(9, 6.5), dpi=300)
    fig = plt.gcf()
    ax_pmf, ax_cdf = fig.axes
    # 保留模板样式，进一步压缩上图留白，上下绘图区高度比为 5:4。
    ax_pmf.get_subplotspec().get_gridspec().set_height_ratios([1.25, 1])
    # 替换模板的数据图元：PMF 使用等宽柱，CDF 使用无填充的右连续阶梯线。
    for ax in (ax_pmf, ax_cdf):
        for artist in list(ax.lines) + list(ax.collections):
            artist.remove()
    ax_pmf.bar(costs, probabilities, width=260, color='C0', alpha=0.85,
               edgecolor='C0', linewidth=1.3, zorder=9)
    ax_pmf.set_ylabel('本档概率', weight='bold', size=12)
    ax_pmf.yaxis.set_major_formatter(PercentFormatter(1))
    for i, (cost, probability) in enumerate(zip(costs, probabilities)):
        ax_pmf.annotate(f'第{i + 1}抽\n{probability:.3%}', (cost, probability),
                        xytext=(0, 36 if i == 1 else 7), textcoords='offset points',
                        ha='center', va='bottom', size=10, weight='bold', color='gray',
                        path_effects=stroke_white, zorder=11)
    ax_cdf.step(np.r_[distribution_plot.x_left_lim, costs, distribution_plot.x_right_lim],
                np.r_[0, cumulative, cumulative[-1]], where='post', color='C0',
                linewidth=1.5, path_effects=stroke_white, zorder=10)
    ax_cdf.scatter(costs, cumulative, s=20, color='C0',
                   edgecolors='white', linewidths=0.8, zorder=11)
    # 前几档星琼间距较小，错开标注并用引线指向其累计概率。
    label_offsets = [(0, 15), (0, 60), (0, 32)] + [(0, 15)] * 4 + [(-8, -10)]
    for i, (cost, probability, offset) in enumerate(zip(costs, cumulative, label_offsets)):
        ax_cdf.annotate(f'{cost}星琼\n{probability:.2%}', (cost, probability),
                        xytext=offset, textcoords='offset points',
                        ha='right' if i == len(costs) - 1 else 'center',
                        va='top' if i == len(costs) - 1 else 'bottom',
                        size=9, weight='bold', color='gray', path_effects=stroke_white,
                        arrowprops={'arrowstyle': '-', 'color': 'gray', 'lw': 0.6}, zorder=12)
    # 主刻度标出实际花费档位，次网格按固定 100 星琼间隔绘制。
    for ax in fig.axes:
        ax.set_xticks(costs)
        ax.set_xticklabels([str(cost) for cost in costs], rotation=35, ha='right')
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.grid(visible=True, axis='x', which='minor', color='lightgray', linewidth=0.5)
        # 期望和等效票数合并至左上角说明，不再在曲线旁重复标注。
        for label in ax.texts:
            if label.get_text().startswith(distribution_plot.text_head):
                label.set_text(label.get_text().replace(
                    f'获取一个{distribution_plot.item_name}期望为{jade_dist.exp:.2f}星琼',
                    f'期望{jade_dist.exp:.2f}星琼，等效{equivalent_warps:.2f}张星轨专票',
                ))
                label.set_y(0.82)
    fig.savefig(output / f'{distribution_plot.title}.png', dpi=300)
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'phantasia_showstage_star_lotto.{suffix}', dpi=300)

    print('活动抽取次数  累计星琼  本次命中概率  累计获取概率')
    for pull, (cost, probability, cdf) in enumerate(zip(costs, probabilities, cumulative), 1):
        print(f'{pull:12d}  {cost:8d}  {probability:12.8%}  {cdf:12.8%}')
    print(f'期望：{jade_dist.exp:.2f}星琼，等效{equivalent_warps:.2f}张星轨专票（160星琼/张）')
    print(f'图像保存至：{output.resolve()}')
    return fig


if __name__ == '__main__':
    plt.close(plot_phantasia_showstage())
