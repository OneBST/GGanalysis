'''
    蓝色星原：旅谣 概率模型推测工具

    基于 GGanalysis.ReverseEngineering.gacha_model_autocracker 的线性软保底自动
    解析工具，结合官方公示的约束反推最可能的概率模型。

    五星（基础0.8%、第71抽开始上升、第90抽硬保底、综合1.58%）：
        1. 线性上升存在下界：只要第71抽起递增且第90抽升至100%，综合概率必然
           >= 1.7633%，与公示的 1.58% 矛盾；完全不设软保底则只有 1.5544%。
        2. 故软保底只能极平缓，且阶梯/线性/等比三种形状都能复现 1.58%，
           单凭综合概率无法区分，需用「第90抽硬吃保底的比例」的实测值判定。
        3. gacha_model.py 默认取其中的线性 +0.167%/抽。

    四星（基础6%、10抽保底、综合公示12%）：
        按官方描述的「五星优先于四星、五星不重置四星保底计数但把四星挤到
        下一抽」精确建模，四星综合概率为 12.9366%，明显高于公示的 12%，
        该公示值无法用「基础6% + 10抽保底」复现。模型本身已用原神参数校验
        通过（五星 1.6052%、四星 13.0016%，对应官方 1.6%/13%）。

    运行方式： python -m GGanalysis.games.azur_promilia.pity_inference
'''
import numpy as np
from GGanalysis.distribution_1d import *
from GGanalysis.games.azur_promilia.gacha_model import *
from GGanalysis.ReverseEngineering.gacha_model_autocracker import LinearAutoCracker

__all__ = [
    'AVG_P', 'AVG_P_4STAR',
    'pity_p2stats', 'candidate_models', 'print_model_table',
    'linear_autocracker_report', 'four_star_report',
    'exact_four_star_rate', 'required_step', 'required_flat_p', 'required_four_star_base',
]

AVG_P = 0.0158        # 公示五星综合概率
AVG_P_4STAR = 0.12    # 公示四星综合概率


def _linear_pity(step):
    '''线性递增：第71抽起每抽 +step，第90抽硬保底'''
    pity_p = linear_p_increase(base_p=BASE_P, pity_begin=SOFT_PITY_BEGIN, step=step, hard_pity=HARD_PITY)
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
    pity_p[SOFT_PITY_BEGIN:HARD_PITY] = BASE_P * ratio ** np.arange(HARD_PITY-SOFT_PITY_BEGIN)
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
        '线性 +0.167%/抽(默认)': PITY_5STAR,
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


def exact_four_star_rate(pity5_p=None, pity4_p=None):
    '''精确计算四星综合概率，纳入「五星优先于四星、五星不重置四星保底计数但把四星挤到下一抽」

    对 (五星保底计数, 四星保底计数) 联合马尔可夫链求平稳分布后累加四星出货率。
    '''
    if pity5_p is None:
        pity5_p = PITY_5STAR
    if pity4_p is None:
        pity4_p = PITY_4STAR
    n5_num, n4_num = len(pity5_p)-1, len(pity4_p)-1
    idx = lambda n, m: n*n4_num + m
    trans = np.zeros((n5_num*n4_num, n5_num*n4_num))
    for n in range(n5_num):
        p5 = pity5_p[n+1]
        for m in range(n4_num):
            p4 = pity4_p[m+1]
            i = idx(n, m)
            trans[i, idx(0, min(m+1, n4_num-1))] += p5                              # 出五星，四星被挤到下一抽
            trans[i, idx(min(n+1, n5_num-1), 0)] += (1-p5) * p4                     # 未出五星，出四星
            trans[i, idx(min(n+1, n5_num-1), min(m+1, n4_num-1))] += (1-p5)*(1-p4)  # 都没出
    mat = trans.T - np.eye(n5_num*n4_num)
    mat[-1] = 1
    rhs = np.zeros(n5_num*n4_num)
    rhs[-1] = 1
    stat = np.linalg.solve(mat, rhs)
    return float(sum(stat[idx(n, m)] * (1-pity5_p[n+1]) * pity4_p[m+1]
                     for n in range(n5_num) for m in range(n4_num)))


def four_star_report():
    '''检验四星公示信息与「基础6% + 10抽保底」机制的一致性'''
    print('=' * 70)
    print('四、四星模型一致性检验')
    print('=' * 70)
    # 先用原神参数校验交互模型实现是否正确
    from GGanalysis.games.genshin_impact.gacha_model import PITY_5STAR as G5, PITY_4STAR as G4
    print(f'模型校验（原神参数）: 五星={1/p2exp(G5)*100:.4f}% (官方1.6%)  '
          f'四星={exact_four_star_rate(G5, G4)*100:.4f}% (官方13%)')
    print()
    naive, exact = 1/p2exp(PITY_4STAR), exact_four_star_rate()
    print(f'近似模型（不考虑五星干扰）  : {naive*100:.4f}%')
    print(f'精确模型（含五星挤压交互）  : {exact*100:.4f}%')
    print(f'官方公示综合概率            : {AVG_P_4STAR*100:.4f}%')
    print(f'差距                        : {(exact-AVG_P_4STAR)*100:.4f} 个百分点')
    print()
    print(f'反推：若综合概率确为 12%，在 10 抽保底不变的前提下基础概率需为 '
          f'{required_four_star_base()*100:.4f}%（而非公示的 {BASE_P_4STAR*100:.1f}%）')
    print('说明：原神式交互模型已用原神官方四星 13% 校验通过，机制理解无误；')
    print('      因此 12% 与「基础6% + 10抽保底」彼此不相容，二者至少有一项需修正。')
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


def required_step(target_avg=AVG_P, pity_begin=SOFT_PITY_BEGIN):
    '''求解线性上升下达成目标综合概率所需的每抽递增值'''
    return _bisect_solve(lambda step: 1/p2exp(_linear_pity(step))-target_avg,
                         1e-6, (1-BASE_P)/(HARD_PITY-pity_begin+1))


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
    linear_autocracker_report()

    print('=' * 70)
    print('二、与公示综合概率 1.58% 相符的五星上升形状')
    print('=' * 70)
    print(f'  阶梯型恒定概率  p = {required_flat_p()*100:.4f}%')
    print(f'  线性型递增值    + {required_step()*100:.4f}% / 抽  （默认取 +0.167%/抽）')
    print(f'  等比型公比      × 1.132')
    print()
    for name, pity_p in [('阶梯', _flat_pity(required_flat_p())), ('线性', PITY_5STAR),
                         ('等比', _geometric_pity(1.132))]:
        print(f'  {name} 71~89抽概率: ' + ' '.join(f'{p*100:.2f}' for p in pity_p[71:90]))
    print()

    print('=' * 70)
    print('三、五星候选模型统计量对照')
    print('=' * 70)
    print_model_table()
    print()
    print('结论：公示的综合概率 1.58% 已排除「原神式线性升满」——该模型必然 >= 1.7633%。')
    print('      最可能的模型是软保底仅轻微抬升（71~89抽约 1.0%~4.0%）。')
    print('      下一步优先统计第90抽硬吃保底的实际比例：')
    print('        若接近 49% -> 基本没有软保底，只剩硬保底；')
    print('        若接近 39% -> 阶梯型（71~89抽恒定为 1.93%）；')
    print('        若接近 35% -> 线性型（71抽起每抽 +0.167%）；')
    print('        若接近 32% -> 等比型（71抽起每抽 ×1.132）。')
    print()

    four_star_report()
