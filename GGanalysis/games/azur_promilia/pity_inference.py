"""三测候选机制诊断；曲线形状和四星10抽保底均为假设，综合概率不能唯一确定机制。"""
import numpy as np
from scipy.optimize import brentq
from GGanalysis.games.azur_promilia.stationary_p import AVG_P_4STAR, exact_four_star_rate
from GGanalysis.distribution_1d import calc_expectation, linear_p_increase, p2dist, p2exp
from GGanalysis.games.azur_promilia.gacha_model import (
    BASE_P, SOFT_PITY_BEGIN, HARD_PITY, HARD_PITY_4STAR,
    PITY_5STAR, PITY_W5STAR,
    HARD_PITY_W5STAR, BASE_P_W5STAR, SOFT_PITY_BEGIN_W5STAR,
)
from GGanalysis.reverse_engineering.gacha_model_autocracker import LinearAutoCracker

# 综合概率仅用于反推目标，不参与抽卡模型的定义。
AVG_P = 0.0158
AVG_P_W5STAR = 0.0209

__all__ = [
    'AVG_P', 'AVG_P_4STAR',
    'pity_p2stats', 'candidate_models', 'print_model_table',
    'linear_autocracker_report',
    'required_step', 'required_flat_p', 'required_four_star_base',
]


def _fit_linear_pity(base: float, avg: float, begin: int, hard: int) -> tuple[np.ndarray, float]:
    """固定起涨点，拟合线性增量；硬保底单独设为1，不提前保底。"""
    def table(step: float) -> np.ndarray:
        p = np.zeros(hard + 1)
        p[1:begin] = base
        p[begin:hard] = base + np.arange(1, hard - begin + 1) * step
        p[hard] = 1
        return p

    step = brentq(lambda s: 1 / p2exp(table(s)) - avg,
                  0, (1 - base) / (hard - begin), xtol=1e-14)
    return table(step), step


def _linear_pity(step, pity_begin=SOFT_PITY_BEGIN):
    '''线性递增：第71抽起每抽 +step，第90抽硬保底'''
    pity_p = linear_p_increase(base_p=BASE_P, pity_begin=pity_begin, step=step, hard_pity=HARD_PITY)
    pity_p[HARD_PITY] = 1
    return pity_p


def _flat_pity(soft_p):
    '''阶梯型：71~89抽恒定为 soft_p'''
    pity_p = np.zeros(HARD_PITY+1)
    pity_p[1:SOFT_PITY_BEGIN] = BASE_P
    pity_p[SOFT_PITY_BEGIN:HARD_PITY] = soft_p
    pity_p[HARD_PITY] = 1
    return pity_p


def _geometric_pity(ratio):
    '''等比递增：第71抽起每抽概率乘以 ratio'''
    pity_p = np.zeros(HARD_PITY+1)
    pity_p[1:SOFT_PITY_BEGIN] = BASE_P
    pity_p[SOFT_PITY_BEGIN:HARD_PITY] = BASE_P * ratio ** np.arange(1, HARD_PITY-SOFT_PITY_BEGIN+1)
    pity_p[HARD_PITY] = 1
    return pity_p


def pity_p2stats(pity_p, hard_pity=HARD_PITY):
    '''计算概率提升表对应的综合概率、期望、硬吃保底比例等统计量'''
    dist = np.asarray(p2dist(pity_p).dist, dtype=float)
    if len(dist) < hard_pity+1:
        dist = np.pad(dist, (0, hard_pity+1-len(dist)))
    return {
        'avg_p': 1/calc_expectation(dist),
        'exp': calc_expectation(dist),
        'hard_pity_rate': dist[hard_pity],   # 硬吃硬保底的比例
        'p_80': np.cumsum(dist)[80],         # 80抽内出货比例
    }


def candidate_models():
    '''返回待比对的候选概率表，键名为模型描述'''
    return {
        '无软保底(仅硬保底)': _flat_pity(BASE_P),
        '阶梯 恒定1.93%': _flat_pity(required_flat_p()),
        '线性拟合(默认)': PITY_5STAR,
        '等比 ×1.132': _geometric_pity(1.132),
        '线性 +1%/抽': _linear_pity(0.01),
        '线性 +2%/抽': _linear_pity(0.02),
        '线性 升至90抽满(原神式)': _linear_pity((1-BASE_P)/(HARD_PITY-SOFT_PITY_BEGIN+1)),
    }


def print_model_table(models=None):
    '''打印候选模型的统计量对照表'''
    if models is None:
        models = candidate_models()
    print(f'{"候选模型":<24s}{"综合概率":>10s}{"期望抽数":>10s}{"第90抽保底率":>13s}{"80抽内出货":>12s}')
    print('-' * 70)
    for name, pity_p in models.items():
        s = pity_p2stats(pity_p)
        flag = ' *' if abs(s['avg_p']-AVG_P) < 1e-4 else ''
        print(f'{name:<24s}{s["avg_p"]*100:>9.4f}%{s["exp"]:>10.2f}'
              f'{s["hard_pity_rate"]*100:>12.2f}%{s["p_80"]*100:>11.2f}%{flag}')
    print('-' * 70)
    print(f'* 表示与公示综合概率 {AVG_P*100:.2f}% 相符的模型')


def linear_autocracker_report():
    '''使用 gacha_model_autocracker 解析线性上升模型'''
    print('=' * 70)
    print('一、用 gacha_model_autocracker 解析：线性上升模型')
    print('=' * 70)
    cracker = LinearAutoCracker(BASE_P, AVG_P, HARD_PITY, pity_begin=SOFT_PITY_BEGIN)
    upper, lower = cracker.search_params(step_value=0.0001)
    print(f'固定 pity_begin={SOFT_PITY_BEGIN}（公示值）：')
    print(f'  高于目标的最接近解: begin={upper[1]}, step={upper[2]*100:.4f}%/抽, 综合概率={upper[3]*100:.4f}%')
    if lower[1] == -1:
        print('  低于目标的最接近解: 不存在 —— 该起始位置下综合概率有下界，取最小步长时仍高于 1.58%')
    print()
    forced = LinearAutoCracker(BASE_P, AVG_P, HARD_PITY, pity_begin=SOFT_PITY_BEGIN, forced_hard_pity=True)
    f_upper, f_lower = forced.search_params(step_value=0.0001)
    print(f'固定 pity_begin={SOFT_PITY_BEGIN} 且允许硬保底单独置 1：')
    print(f'  高于目标的最接近解: step={f_upper[2]*100:.4f}%/抽, 综合概率={f_upper[3]*100:.4f}%')
    print(f'  低于目标的最接近解: step={f_lower[2]*100:.4f}%/抽, 综合概率={f_lower[3]*100:.4f}%')
    print()
    free = LinearAutoCracker(BASE_P, AVG_P, HARD_PITY)
    n_upper, n_lower = free.search_params(step_value=0.001)
    print('不固定 pity_begin（仅作对照，已偏离公示的第71抽）：')
    print(f'  高于目标的最接近解: begin={n_upper[1]}, step={n_upper[2]*100:.3f}%/抽, 综合概率={n_upper[3]*100:.4f}%')
    print(f'  低于目标的最接近解: begin={n_lower[1]}, step={n_lower[2]*100:.3f}%/抽, 综合概率={n_lower[3]*100:.4f}%')
    print()


def _bisect_solve(f, low, high):
    '''二分法求解 f(x)=0，避免引入 scipy 依赖'''
    for _ in range(200):
        mid = (low + high) / 2
        if f(mid) > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def required_step(target_avg: float = AVG_P, pity_begin: int = SOFT_PITY_BEGIN) -> float:
    """返回角色池达到target_avg的线性增量；pity_begin为首次递增抽数。"""
    return _fit_linear_pity(BASE_P, target_avg, pity_begin, HARD_PITY)[1]

def required_flat_p(target_avg=AVG_P):
    '''求解阶梯型下达成目标综合概率所需的恒定概率'''
    return _bisect_solve(lambda soft_p: 1/p2exp(_flat_pity(soft_p))-target_avg, BASE_P, 0.2)


def required_four_star_base(target_avg=AVG_P_4STAR, hard_pity=HARD_PITY_4STAR):
    '''在精确交互模型下，反推达成目标四星综合概率所需的基础概率'''
    def diff(base_p):
        pity4 = np.zeros(hard_pity+1)
        pity4[1:hard_pity] = base_p
        pity4[hard_pity] = 1
        return exact_four_star_rate(PITY_5STAR, pity4) - target_avg
    return _bisect_solve(diff, 0.001, 0.2)


if __name__ == '__main__':
    for name, base, avg, hard, begin, table in (
        ('角色', BASE_P, AVG_P, HARD_PITY, SOFT_PITY_BEGIN, PITY_5STAR),
        ('武器', BASE_P_W5STAR, AVG_P_W5STAR, HARD_PITY_W5STAR, SOFT_PITY_BEGIN_W5STAR, PITY_W5STAR),
    ):
        cracker = LinearAutoCracker(base, avg, hard, pity_begin=begin, forced_hard_pity=True)
        print(name, '反推网格上下界', cracker.search_params(step_value=0.0001))
        fitted, step = _fit_linear_pity(base, avg, begin, hard)
        print(name, '拟合增量', step, '与模型概率表的最大差', np.max(np.abs(fitted-table)))
        print(name, pity_p2stats(fitted, hard))
