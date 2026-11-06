"""三测两池的稳态概率；四星10抽保底及五星重置规则仍为待验证假设。

运行：python -m GGanalysis.games.azur_promilia.stationary_p
"""
from GGanalysis.games.azur_promilia.gacha_model import (
    PITY_5STAR, PITY_W5STAR, PITY_4STAR,
)
from GGanalysis.distribution_1d import p2exp
from GGanalysis.markov import PriorityPityChain

# 公示综合概率，仅用于比较稳态结果与反推基础概率。
AVG_P_4STAR = 0.12

__all__ = ['exact_four_star_rate', 'four_star_report']

def exact_four_star_rate(pity5_p=None, pity4_p=None, reset_by_five_star=False):
    """返回假设规则下四星稳态概率，默认五星不重置四星。

    两张概率表默认取角色池。6%视为无条件概率，实际四星概率为
    min(p4, 1-p5)，修正旧实现将6%当成未出五星时条件概率的口径。
    """
    chain = PriorityPityChain(
        [PITY_5STAR if pity5_p is None else pity5_p,
         PITY_4STAR if pity4_p is None else pity4_p],
        extra_state=0, remove_pity=reset_by_five_star)
    return float(chain.stationary_rates()[1])


def four_star_report() -> None:
    """直接复用预置保底链，对比五星是否重置四星时的综合概率。"""
    print('独立四星10抽保底综合概率', 1/p2exp(PITY_4STAR))
    for name, pity5 in (('角色池', PITY_5STAR), ('武器池', PITY_W5STAR)):
        for reset in (False, True):
            system = PriorityPityChain([pity5, PITY_4STAR, [0, 1]], remove_pity=reset)
            rates = system.stationary_rates()
            print(name, '五星重置四星', reset, '各星级综合概率', rates,
                  '四星与12%的差', rates[1]-AVG_P_4STAR)
    print('基础6%、综合12%不足以唯一反推四星保底规则；10抽保底仍为旧版假设。')


if __name__ == '__main__':
    four_star_report()
