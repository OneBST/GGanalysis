"""活动盲盒花费分布：从仓库根目录运行 python -m GGanalysis.games.neverness_to_everness.figure_plot_blind_box。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

from GGanalysis.gacha_plot import DrawDistribution
from GGanalysis.games.neverness_to_everness.gacha_data import BLIND_BOX_CONFIGS
from GGanalysis.games.neverness_to_everness.gacha_model import blind_boxes
from GGanalysis.plot_tools import stroke_white


def plot_blind_box(version: str = "1.1", target: int = 0,
                   save_path: str | Path = "figure") -> Figure:
    """沿用双图模板绘制首次获得指定奖品的环石花费，保存 PNG/SVG 并打印各档概率。

    version 为 blind_boxes 的期次键，target 为该期 prizes 的索引；默认主载具。
    从完整奖池开始，命中目标即停止；方斯期望另按全程用方斯支付计算。
    """
    model = blind_boxes[version]
    config = BLIND_BOX_CONFIGS[version]
    means = model.expectations(target)
    pull_dist = model(target)
    pulls = np.flatnonzero(pull_dist.dist)
    costs = np.r_[0, np.cumsum(model.costs["annulith"])][pulls]
    probabilities = pull_dist.dist[pulls]
    cumulative = np.cumsum(probabilities)
    output = Path(save_path)
    output.mkdir(parents=True, exist_ok=True)
    prize = config["prizes"][target]
    description = (
        "采用资料汇总权重与价格，从完整奖池开始\n"
        "加权不放回抽取，获得目标即停止\n"
        f"期望 {means['annulith']:.2f} 环石\n"
        f"全用方斯支付：期望 {means['fons_10k']:.2f} 万方斯\n"
        f"平均开盲盒 {means['pull']:.2f} 次，最多 {model.total} 次\n"
        "两种货币分别计算，未扣奖励返还\n"
        "@一棵平衡树 2026-10-31"
    )
    plot = DrawDistribution(
        dist_data=model(target, unit="annulith"),
        title=f"异环{version}「{config['name']}」盲盒花费分布",
        item_name=prize, cost_name="环石", save_path=str(output),
        quantile_pos=[], is_finite=True, show_peak=False, show_exp=False,
        description_func=lambda **kwargs: description,
    )
    plot.switch_len = len(plot.dist_data)
    padding = max(250, costs[-1] * 0.04)
    plot.x_left_lim, plot.x_right_lim = -padding, costs[-1] + padding
    # 先用项目现有模板布置标题、网格和双图，再替换为离散柱和无填充 CDF。
    plot.show_two_graph(savefig=True, figsize=(9, 6.5), dpi=300)
    fig = plt.gcf()
    ax_pmf, ax_cdf = fig.axes
    ax_pmf.get_subplotspec().get_gridspec().set_height_ratios([1.25, 1])
    ax_pmf.set_title(f"目标：{prize}", weight="bold", size=12)
    for ax in fig.axes:
        for artist in list(ax.lines) + list(ax.collections):
            artist.remove()
        ax.set_xticks(costs)
        ax.set_xticklabels([str(cost) for cost in costs], rotation=35, ha="right")
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.grid(visible=True, axis="x", which="minor", color="lightgray", linewidth=0.5)
        for label in ax.texts:
            if label.get_text() == description:
                label.set_transform(ax.transAxes)
                label.set_position((0.03, 0.82))
                label.set_va("top")
                label.set_fontsize(10)
    width = min(np.diff(costs)) * 0.65 if len(costs) > 1 else padding
    ax_pmf.bar(costs, probabilities, width=width, color="C0", alpha=0.85,
               edgecolor="C0", linewidth=1.3, zorder=9)
    ax_pmf.set_ylim(0, probabilities.max() * 1.22)
    ax_pmf.set_ylabel("本档概率", weight="bold", size=12)
    ax_pmf.yaxis.set_major_formatter(PercentFormatter(1))
    for i, (pull, cost, probability) in enumerate(zip(pulls, costs, probabilities)):
        ax_pmf.annotate(f"第{pull}抽\n{probability:.2%}", (cost, probability),
                        xytext=(0, 7 + (24 if i < 3 and i % 2 else 0)),
                        textcoords="offset points", ha="center", va="bottom",
                        size=9, weight="bold", color="gray", path_effects=stroke_white)
    ax_cdf.step(np.r_[plot.x_left_lim, costs, plot.x_right_lim],
                np.r_[0, cumulative, cumulative[-1]], where="post", color="C0",
                linewidth=1.5, path_effects=stroke_white, zorder=10)
    ax_cdf.scatter(costs, cumulative, s=20, color="C0", edgecolors="white", zorder=11)
    ax_cdf.set_ylim(0, 1.12)
    for i, (cost, probability) in enumerate(zip(costs, cumulative)):
        last = i == len(costs) - 1
        ax_cdf.annotate(f"{cost}环石\n{probability:.2%}", (cost, probability),
                        xytext=(-8, -12) if last else (0, 12 + 30 * (i % 2)),
                        textcoords="offset points", ha="right" if last else "center",
                        va="top" if last else "bottom", size=8, weight="bold",
                        color="gray", path_effects=stroke_white,
                        arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.6}, zorder=12)
    fig.savefig(output / f"{plot.title}.png", dpi=300)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"nte_blind_box_{version}_{target}.{suffix}", dpi=300)
    print(f"{version} {prize}：{means['annulith']:.2f} 环石 / "
          f"{means['fons_10k']:.2f} 万方斯；开盲盒 {means['pull']:.2f} 次")
    print("抽次  累计环石  本档概率  累积概率")
    for pull, cost, probability, cdf in zip(pulls, costs, probabilities, cumulative):
        print(f"{pull:2d}  {cost:6d}  {probability:.6%}  {cdf:.6%}")
    return fig


if __name__ == "__main__":
    for version in blind_boxes:
        plt.close(plot_blind_box(version))
