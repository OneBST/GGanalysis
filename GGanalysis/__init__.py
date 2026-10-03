'''
    抽卡游戏抽卡概率计算工具包 GGanalysis
    by 一棵平衡树OneBST
'''
from importlib import import_module as _import_module

from . import distribution_1d as _distribution_1d

# 父包总会先于子模块执行。这里按需导出，避免仅导入 FiniteDist 时
# 连带初始化抽卡层、状态分布和马尔可夫工具。
_EXPORT_GROUPS = {
    'distribution_1d': _distribution_1d.__all__,
    'state_distribution': ('StateDist', 'StateKernel', 'stateful_item_num_dist'),
    'gacha_layers': (
        'comb', 'binom', 'warnings', 'GachaLayer', 'PityLayer', 'BernoulliLayer',
        'MarkovLayer', 'DynamicProgrammingLayer', 'CouponCollectorLayer',
    ),
    'basic_models': (
        'GachaModel', 'CommonGachaModel', 'BernoulliGachaModel', 'CouponCollectorModel',
        'PityCouponCollectorModel', 'DualPityCouponCollectorModel',
        'GeneralCouponCollectorModel', 'PityModel', 'DualPityModel',
        'PityBernoulliModel', 'DualPityBernoulliModel',
    ),
    'markov.coupon_collection': (
        'GeneralCouponCollection', 'lru_cache', 'random', 'get_equal_coupon_collection_exp',
    ),
    'markov.priority_pity': ('PriorityPityChain', 'stationary_item_count_distribution'),
    'markov.state_space': ('Selector', 'StateSpace'),
    'markov.transition': ('MarkovTransition',),
    'markov.builder': ('ProbabilityMatrixBuilder', 'TransitionBuilder'),
    'markov.hit_process': ('HitTransitionBuilder', 'HitProcess', 'HitProcessAnalysis'),
    'markov.analysis': (
        'first_hitting_time', 'stationary_power', 'stationary_eigs',
        'stationary_solve', 'StationaryInfo', 'StateRewards', 'group_mass',
        'ChainAnalysis', 'AbsorptionResult', 'TransitionRewards', 'event_count_state_dist',
    ),
}
_EXPORT_MODULES = {
    name: module for module, names in _EXPORT_GROUPS.items() for name in names
}
_SUBMODULES = ('distribution_1d', 'state_distribution', 'gacha_layers', 'basic_models', 'markov')
__all__ = [*_EXPORT_MODULES, *_SUBMODULES]


def __getattr__(name: str) -> object:
    if name in _EXPORT_MODULES:
        module = _import_module('.' + _EXPORT_MODULES[name], __name__)
        value = getattr(module, name)
    elif name in _SUBMODULES:
        value = _import_module('.' + name, __name__)
    else:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
