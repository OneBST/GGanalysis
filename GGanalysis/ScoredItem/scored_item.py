from GGanalysis.distribution_1d import *
from itertools import permutations, combinations
from copy import deepcopy
import time

'''
    词条评分型道具类
    参考 ideless 的思想，将难以处理的多个不同类别的属性，通过线性加权进行打分变成一维问题
    本模块中部分计算方法在ideless的基础上重新实现并抽象
    ideless 原仓库见 https://github.com/ideless/reliq
'''

class ScoredItem():
    '''词条型道具'''
    # TODO 研究怎样继承 stats_score 比较好，运算时 stats_score 应该一致才能运算 套装 stats_score 也应该一致
    def __init__(self, score_dist: Union[FiniteDist, np.ndarray, list]=FiniteDist([0]), sub_stats_exp: dict={}, drop_p=1, stats_score: dict={}) -> None:
        '''使用分数分布和副词条期望完成初始化，可选副词条方差'''
        self.score_dist = FiniteDist(score_dist)    # 得分分布
        self.stats_score = stats_score
        self.sub_stats_exp = sub_stats_exp          # 每种副词条的期望
        self.drop_p = drop_p
        if self.drop_p < 0 or self.drop_p > 1:
            raise ValueError("drop_p should between 0 and 1!")
        self.fit_sub_stats()
        self.null_mark = self.is_null()

    def fit_sub_stats(self):
        '''调整副词条平均值长度以适应分数分布长度'''
        for key in self.sub_stats_exp.keys():
            if len(self.sub_stats_exp[key]) > self.__len__():
                self.sub_stats_exp[key] = self.sub_stats_exp[key][:self.__len__()]
            else:
                self.sub_stats_exp[key] = pad_zero(self.sub_stats_exp[key], self.__len__())

    def is_null(self):
        '''判断自身是否为空'''
        return np.sum(self.score_dist.dist) == 0

    # 以下模块被 scored_item_tools.get_mix_dist 调用
    def sub_stats_clear(self):
        '''清除score_dist为0对应的sub_stats_exp'''
        mask = self.score_dist.dist != 0
        for key in self.sub_stats_exp.keys():
            self.sub_stats_exp[key] = self.sub_stats_exp[key] * mask

    def check_subexp_sum(self) -> np.ndarray:
        '''检查副词条的加权和'''
        ans = np.zeros(len(self))
        for key in self.sub_stats_exp.keys():
            ans[:len(self.sub_stats_exp[key])] += self.sub_stats_exp[key] * self.stats_score.get(key, 0)
        return ans

    def repeat(self, n: int=1, p=None) -> 'ScoredItem':
        '''重复n次获取道具尝试，每次有p概率获得道具后获得的最大值分布'''
        if n == 0:
            return ScoredItem([1], stats_score=self.stats_score)
        if p is None:
            use_p = self.drop_p
        else:
            if p < 0 or p > 1:
                raise ValueError("p should between 0 and 1!")
            use_p = p
        cdf = (use_p * np.cumsum(self.score_dist.dist) + 1 - use_p) ** n
        return ScoredItem(cdf2dist(cdf), self.sub_stats_exp, stats_score=self.stats_score)
    
    def __getattr__(self, key):  # 访问未计算的属性时进行计算
        # 基本统计属性
        if key in ['exp', 'var']:
            if key == 'exp':
                self.exp = self.score_dist.exp
                return self.exp
            if key == 'var':
                self.var = self.score_dist.var
                return self.var
        if key == 'stats_score':    # 词条得分，其值应位于0-1之间，默认为空
            return {}

    def __getitem__(self, sliced):
        '''根据sliced形成遮罩，返回sliced选中区间的值，其他位置设置为0'''
        mask = np.zeros_like(self.score_dist.dist)
        mask[sliced] = 1
        score_dist = np.trim_zeros(self.score_dist.dist * mask, 'b')
        sub_stats_exp = {}
        for key in self.sub_stats_exp.keys():
            sub_stats_exp[key] = self.sub_stats_exp[key] * mask
        return ScoredItem(FiniteDist(score_dist), sub_stats_exp, stats_score=self.stats_score)
    
    def __add__(self, other: 'ScoredItem') -> 'ScoredItem':
        '''数量合并两个物品'''
        # 判断 stats_score 是否一致
        if self.stats_score != other.stats_score:
            raise ValueError("stats_score must be the same!")
        key_set = set(self.sub_stats_exp.keys())
        key_set.update(other.sub_stats_exp.keys())
        ans_dist = self.score_dist + other.score_dist
        ans_sub_stats_exp = {}
        target_len = max(len(self), len(other))
        for key in key_set:
            s_exp = self.sub_stats_exp.get(key, np.zeros(1))
            o_exp = other.sub_stats_exp.get(key, np.zeros(1))
            # 此处要对副词条取平均值
            a = pad_zero(s_exp * self.score_dist.dist, target_len) + \
                pad_zero(o_exp * other.score_dist.dist, target_len)
            b = ans_dist.dist[:target_len]
            ans_sub_stats_exp[key] = np.divide(a, b, \
                                    out=np.zeros_like(a), where=b!=0)
        return ScoredItem(ans_dist, ans_sub_stats_exp, stats_score=self.stats_score)

    def __mul__(self, other: Union['ScoredItem', float, int]) -> 'ScoredItem':
        '''对两个物品进行卷积合并，或单纯数乘'''
        if isinstance(other, ScoredItem):
            # 判断 stats_score 是否一致
            if self.stats_score != other.stats_score:
                raise ValueError("stats_score must be the same!")
            # 两者都是词条型道具的情况下，进行卷积合并
            new_score_dist = self.score_dist * other.score_dist
            new_sub_exp = {}
            key_set = set(self.sub_stats_exp.keys())
            key_set.update(other.sub_stats_exp.keys())
            for key in key_set:
                a = convolve(self.score_dist.dist * self.sub_stats_exp.get(key, np.zeros(1)), \
                             other.score_dist.dist)
                b = convolve(other.score_dist.dist * other.sub_stats_exp.get(key, np.zeros(1)), \
                             self.score_dist.dist)
                a = pad_zero(a, len(new_score_dist))
                b = pad_zero(b, len(new_score_dist))
                new_sub_exp[key] = np.divide((a + b), new_score_dist.dist, out=np.zeros_like(new_score_dist.dist), where=new_score_dist.dist!=0)
            return ScoredItem(new_score_dist, new_sub_exp, stats_score=self.stats_score)
        else:
            # other为常数的情况下，进行数乘，数乘下不影响副词条平均值
            new_score_dist = self.score_dist * other
            new_sub_exp = self.sub_stats_exp
            return ScoredItem(new_score_dist, new_sub_exp, stats_score=self.stats_score)

    def __rmul__(self, other: Union['ScoredItem', float, int]) -> 'ScoredItem':
        return self * other
    
    def __len__(self) -> int:
        return len(self.score_dist)
    
    def __str__(self) -> str:
        return f"Score Dist {self.score_dist.dist} Sub Exp {self.sub_stats_exp}"

class ScoredItemSet():
    '''由词条型道具组成的套装'''
    def __init__(self, item_set:dict={}) -> None:
        '''由词条型道具构成的套装抽象'''
        self.item_set = item_set

    def add_item(self, item_name:str, item: ScoredItem):
        '''添加名为 item_name 的道具'''
        self.item_set[item_name] = item

    def combine_set(self, select_items: list[str]=None, n=1):
        '''计算获取n次道具后套装中道具的最佳得分分布'''
        # TODO 修改这个函数，目前存在意义很奇怪，为什么可以有输入
        ans = ScoredItem([1], stats_score=self.item_set.values()[0].stats_score)
        if select_items is None:
            for key in self.item_set.keys():
                ans *= self.item_set[key].repeat(n, self.item_set[key].drop_p)
        else:
            for key in select_items:
                ans *= self.item_set[key].repeat(n)
        return ans

    def repeat(self, n, p: dict=None) -> list[ScoredItem]:
        '''重复n次获取道具尝试，返回重复后的道具组'''
        ans = []
        for key in sorted(list(self.item_set.keys()), key=str.lower):
            if p is None:
                use_p = self.item_set[key].drop_p
            else:
                use_p = p[key]
                if use_p < 0 or use_p > 1:
                    raise ValueError("use_p should between 0 and 1!")
            ans.append(self.item_set[key].repeat(n, use_p))
        return ans
    
    def to_list(self):
        '''返回以列表形式储存的道具'''
        return list(self.item_set.values())

if __name__ == '__main__':
    pass