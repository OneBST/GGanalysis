from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from typing import Union

# 所有抽卡模型的基类，目前什么也不干
class GachaModel(object):
    pass

# 基本抽卡类 对每次获取道具是独立事件的抽象
class CommonGachaModel(GachaModel):
    def __init__(self) -> None:
        super().__init__()
        # 初始化抽卡层
        self.layers = []
        # 在本层中定义抽卡层
    
    # 调用本类时运行的函数
    def __call__(self, item_num: int=1, multi_dist: bool=False, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        parameter_list = self._build_parameter_list(*args, **kwds)
        # 如果没有对 _build_parameter_list 进行定义就输入参数，报错
        if args != () and kwds != {} and parameter_list is None:
            raise Exception('Parameters is not defined.')
        # 如果 item_num 为 0，则返回 1 分布
        if item_num == 0:
            return FiniteDist([1])
        # 如果 multi_dist 参数为真，返回抽取 [1, 抽取个数] 个道具的分布列表
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        # 其他情况正常返回
        return self._get_dist(item_num, parameter_list)

    # 用于输入参数解析，生成每层对应参数列表
    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        parameter_list = []
        for i in range(len(self.layers)):
            parameter_list.append([[], {}])
        return parameter_list

    # 输入 [完整分布, 条件分布] 指定抽取个数，返回抽取 [1, 抽取个数] 个道具的分布列表
    def _get_multi_dist(self, end_pos: int, parameter_list: list=None):
        input_dist = self._forward(parameter_list)
        ans_list = [FiniteDist([1]), input_dist[1]]
        for i in range(1, end_pos):
            # 添加新的一层并设定方差与期望
            ans_list.append(ans_list[i] * input_dist[0])
            ans_list[i+1].exp = input_dist[1].exp + input_dist[0].exp * i
            ans_list[i+1].var = input_dist[1].var + input_dist[0].var * i
        return ans_list

    # 返回单个分布
    def _get_dist(self, item_num: int, parameter_list: list=None):
        ans_dist = self._forward(parameter_list)
        ans: FiniteDist = ans_dist[1] * ans_dist[0] ** (item_num - 1)
        ans.exp = ans_dist[1].exp + ans_dist[0].exp * (item_num - 1)
        ans.var = ans_dist[1].var + ans_dist[0].var * (item_num - 1)
        return ans

    # 根据多层信息计算获取物品分布列
    def _forward(self, parameter_list: list=None):
        ans_dist = None
        # 没有输入参数返回默认分布
        if parameter_list is None:
            for layer in self.layers:
                ans_dist = layer(ans_dist)
            return ans_dist
        # 有输入参数则将分布逐层推进
        for parameter, layer in zip(parameter_list, self.layers):
            ans_dist = layer(ans_dist, *parameter[0], **parameter[1])
        return ans_dist

# 伯努利抽卡类
class BernoulliGachaModel(GachaModel):
    def __init__(self, p) -> None:
        super().__init__()
        self.p = p  # 伯努利试验概率

    def __call__(self, item_num: int, calc_pull: int) -> FiniteDist:
        '''
            返回抽物品个数的分布
            这里 calc_pull 表示了计算的最高抽数，高于此不计算
        '''
        x = np.arange(calc_pull+1)
        dist = self.p * (binom.pmf(item_num-1, x-1, self.p))
        dist[0] = 0
        return FiniteDist(dist)

    # 计算固定抽数，获得道具数的分布
    '''
    def _get_dist(self, item_num, pulls):  # 恰好在第x抽抽到item_num个道具的概率，限制长度最高为pulls
        x = np.arange(pulls+1)
        dist = self.p * (binom.pmf(0, x-1, self.p))
        dist[0] = 0
        return finite_dist_1D(dist)
    '''

# 带保底抽卡类
class PityModel(CommonGachaModel):
    def __init__(self, pity_p) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p))
    
    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity = 0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0) -> list:
        parameter_list = [[[], {'item_pity':item_pity}]]
        return parameter_list

# 双重保底抽卡类
class DualPityModel(CommonGachaModel):
    def __init__(self, pity_p1, pity_p2) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p1))
        self.layers.append(PityLayer(pity_p2))

    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity = 0, up_pity = 0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, up_pity, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0, up_pity: int=0) -> list:
        parameter_list = [
            [[], {'item_pity':item_pity}],
            [[], {'item_pity':up_pity}],
        ]
        return parameter_list

# 保底伯努利抽卡类
class PityBernoulliModel(CommonGachaModel):
    def __init__(self, pity_p, p, e_error = 1e-8, max_dist_len=1e5) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p))
        self.layers.append(BernoulliLayer(p, e_error, max_dist_len))
    
    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity=0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0) -> list:
        parameter_list = [
            [[], {'item_pity':item_pity}],
            [[], {}],
        ]
        return parameter_list

# 双重保底伯努利类
class DualPityBernoulliModel(CommonGachaModel):
    def __init__(self, pity_p1, pity_p2, p, e_error = 1e-8, max_dist_len=1e5) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p1))
        self.layers.append(PityLayer(pity_p2))
        self.layers.append(BernoulliLayer(p, e_error, max_dist_len))
    
    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity = 0, up_pity = 0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, up_pity, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0, up_pity: int=0) -> list:
        parameter_list = [
            [[], {'item_pity':item_pity}],
            [[], {'item_pity':up_pity}],
            [[], {}],
        ]
        return parameter_list