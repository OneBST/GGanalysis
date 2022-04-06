from GGanalysisLite.distribution_1d import *
from GGanalysisLite.gacha_layers import *

# 基本抽卡类
class common_gacha_model():
    def __init__(self) -> None:
        # 初始化抽卡层
        self.layers = []
        # 在本层中定义抽卡层
    
    def __call__(self, *args: any, **kwds: any) -> finite_dist_1D:
        parameter_list = self._build_parameter_list(*args, **kwds)
        return self._forward(parameter_list)[1]

    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        return None

    # 输入 [完整分布, 条件分布] 指定抽取个数，返回抽取 [1, 抽取个数] 个道具的分布列表
    def _get_multi_dist(self, end_pos: int, parameter_list: list=None):
        input_dist = self._forward(parameter_list)
        ans_matrix = [finite_dist_1D([1]), input_dist[1]]
        for i in range(1, end_pos):
            # 添加新的一层并设定方差与期望
            ans_matrix.append(ans_matrix[i] * input_dist[0])
            ans_matrix[i+1].exp = input_dist[1].exp + input_dist[0].exp * i
            ans_matrix[i+1].var = input_dist[1].var + input_dist[0].var * i
        return ans_matrix

    def _get_dist(self, item_num: int, parameter_list: list=None):
        ans_dist = self._forward(parameter_list)
        ans: finite_dist_1D = ans_dist[1] * ans_dist[0] ** (item_num - 1)
        ans.exp = ans_dist[1].exp + ans_dist[0].exp * (item_num - 1)
        ans.var = ans_dist[1].var + ans_dist[0].var * (item_num - 1)
        return ans

    def get_dist(self, item_num: int=1, parameter_list: list=None):
        return self._get_dist(item_num, parameter_list)

    def _forward(self, parameter_list: list=None):
        ans_dist = None
        # 没有输入参数返回默认分布
        if parameter_list is None:
            for layer in self.layers:
                ans_dist = layer(ans_dist)
            return ans_dist
        # 有输入参数则将分布逐层推进
        for parameter, layer in zip(parameter_list, self.layers):
            # print(a[1])
            ans_dist = layer(ans_dist, *parameter[0], **parameter[1])
            # self.test(*parameter[0], **parameter[1])
        return ans_dist
