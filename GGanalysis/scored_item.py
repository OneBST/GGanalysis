from GGanalysis.distribution_1d import *
from itertools import permutations, combinations
from copy import deepcopy
import time

'''
    词条强化型道具类
    参考 ideless 的思想，将难以处理的多个不同类别的属性，通过线性加权进行打分变成一维问题
    本模块中部分计算方法在ideless的基础上重新实现并抽象
    ideless 原仓库见 https://github.com/ideless/reliq
'''

class ScoredItem():
    '''词条型道具'''
    def __init__(self, score_dist: Union[FiniteDist, np.ndarray, list]=FiniteDist([0]), sub_stats_exp: dict={}) -> None:
        '''使用分数分布和副词条期望完成初始化，可选副词条方差'''
        self.score_dist = FiniteDist(score_dist)
        self.sub_stats_exp = sub_stats_exp
        self.fit_sub_stats()
        self.null_mark = self.is_null()
        # TODO 给出分布和方差计算，或者直接提取 score_dist 的结果

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

    # def sub_stats_clear(self):
    #     '''清除score_dist为0对应的sub_stats_exp'''
    #     mask = self.score_dist.dist != 0
    #     for key in self.sub_stats_exp.keys():
    #         self.sub_stats_exp[key] = self.sub_stats_exp[key] * mask

    def repeat(self, n: int, p=1) -> 'ScoredItem':
        '''重复n次获取道具尝试，每次有p概率获得道具后获得的最大值分布'''
        if p < 0 or p > 1:
            raise ValueError("p should between 0 and 1!")
        cdf = (p * np.cumsum(self.score_dist.dist) + 1 - p) ** n
        return ScoredItem(FiniteDist(cdf2dist(cdf)), self.sub_stats_exp)
    
    def __getattr__(self, key):  # 访问未计算的属性时进行计算
        # 基本统计属性
        if key in ['exp', 'var']:
            if key == 'exp':
                self.exp = self.score_dist.exp
                return self.exp
            if key == 'var':
                self.var = self.score_dist.var
                return self.var

    def __getitem__(self, sliced):
        '''根据sliced形成遮罩，返回sliced选中区间的值，其他位置设置为0'''
        mask = np.zeros_like(self.score_dist.dist)
        mask[sliced] = 1
        score_dist = np.trim_zeros(self.score_dist.dist * mask, 'b')
        sub_stats_exp = {}
        for key in self.sub_stats_exp.keys():
            sub_stats_exp[key] = self.sub_stats_exp[key] * mask
        return ScoredItem(FiniteDist(score_dist), sub_stats_exp)
    
    def __add__(self, other: 'ScoredItem') -> 'ScoredItem':
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
        return ScoredItem(ans_dist, ans_sub_stats_exp)

    def __mul__(self, other: Union['ScoredItem', float, int]) -> 'ScoredItem':
        '''对两个物品进行卷积合并，或单纯数乘'''
        if isinstance(other, ScoredItem):
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
            return ScoredItem(new_score_dist, new_sub_exp)
        else:
            # other为常数的情况下，进行数乘，数乘下不影响副词条平均值
            new_score_dist = self.score_dist * other
            new_sub_exp = self.sub_stats_exp
            return ScoredItem(new_score_dist, new_sub_exp)

    def __rmul__(self, other: Union['ScoredItem', float, int]) -> 'ScoredItem':
        return self * other
    
    def __len__(self) -> int:
        return len(self.score_dist)
    
    def __str__(self) -> str:
        return f"Score Dist {self.score_dist.dist} Sub Exp {self.sub_stats_exp}"

def combine_items(item_list: list[ScoredItem]):
    '''返回列表内道具的混合'''
    ans = ScoredItem([1], {})
    for item in item_list:
        ans *= item
    return ans

class ScoredItemSet():
    '''由词条型道具组成的套装'''
    def __init__(self, item_set:dict={}, item_p:dict={}) -> None:
        '''由词条型道具构成的套装抽象'''
        self.item_set = item_set
        self.item_p = item_p

    def set_item(self, item_name:str, item: ScoredItem):
        '''添加名为 item_name 的道具'''
        self.item_set[item_name] = item

    def combine_set(self):
        '''混合套装中道具'''
        # TODO 修改为按照传入key进行混合，默认全部一起混合
        ans = ScoredItem([1])
        for key in self.item_set.keys():
            ans *= self.item_set[key]
        return ans

    def repeat(self, n):
        '''重复n次获取道具尝试，返回重复后的道具组'''
        ans = {}
        for key in self.item_set.keys():
            ans[key] = self.item_set[key].repeat(n, self.item_p.get(key, 0))
        return ScoredItemSet(ans)
    
    def item_list(self):
        '''返回以列表形式储存的道具'''
        return list(self.item_set.values())
    
class ConditionalScore():
    '''返回可行的分数序列'''
    def __init__(self, item_idx:Union[list, tuple], score_max:list) -> None:
        self.state = [score_max[0]]
        self.item_idx = item_idx
        self.score_max = score_max
        # 生成初始状态
        for i in range(1, len(self.item_idx)):
            if self.item_idx[i] > self.item_idx[i-1]:
                self.state.append(min(self.score_max[i], self.state[i-1]))
            else:
                self.state.append(min(self.score_max[i], self.state[i-1])-1)
        # 定义合法分数序列集合
        self.possible_sequence = []
        # 枚举并选择合法分数序列
        while self.state[-1] >= 0:
            self.possible_sequence.append(deepcopy(self.state))
            self.next_state()
        
    def next_state(self):
        '''切换到下一个可行状态'''
        pos = len(self.state)
        while pos >= 1:
            pos -= 1
            self.state[pos] -= 1
            # TODO 0也是可能的，可能都是0，但是需要特判，要想一下其他地方怎么写
            if self.state[pos] >= 0:
                break
        # 进行迭代
        for i in range(pos+1, len(self.state)):
            if self.item_idx[i] > self.item_idx[i-1]:
                self.state[i] = min(self.score_max[i], self.state[i-1])
            else:
                self.state[i] = min(self.score_max[i], self.state[i-1])-1

    def __len__(self) -> int:
        return len(self.possible_sequence)
    
    def __getitem__(self, idx):
        return self.possible_sequence[idx]
    
def select_best_combination(item_set:list[ScoredItem], chose_num=1):
    '''返回选取最优chose_num件后的情况，注意复杂度是关于以chose_num为幂次的多项式'''
    '''这个函数只在选择1或者选择2的情况下的容斥处理是对的'''
    # 预处理小于等于某分数的概率
    if chose_num > 2:
        raise ValueError("chose_num should not greater than 2!")
    p_less = [np.cumsum(item.score_dist.dist) for item in item_set]
    max_lenth = max([len(item) for item in item_set]) * chose_num + 1
    ans_dist = np.zeros(max_lenth)
    ans_sub_exp = {}
    start_time = time.time()
    for perm in permutations(range(len(item_set)), chose_num):
        # 枚举所有排列
        score_max = [len(item_set[i])-1 for i in perm]
        all_score_list = ConditionalScore(perm, score_max)
        # print('Computing:', perm, 'Tasks:', len(all_score_list))
        for score_list in all_score_list:
            p = 1
            score = 0
            for i, s in zip(perm, score_list):
                score += s
                p *= item_set[i].score_dist[s]
            for i in range(len(item_set)):
                if i in perm:
                    continue
                if i > perm[-1]:
                    pos = score_list[-1]
                else:
                    pos = score_list[-1]-1
                if pos < 0:
                    p = 0
                    break
                pos = min(pos, len(p_less[i])-1)
                p *= p_less[i][pos]
            if p == 0:
                continue
            ans_dist[score] += p
            for i, s in zip(perm, score_list):
                for key in item_set[i].sub_stats_exp.keys():
                    if key not in ans_sub_exp:
                        ans_sub_exp[key] = np.zeros(max_lenth)
                    ans_sub_exp[key][score] += p * item_set[i].sub_stats_exp[key][s]
    # 求副词条平均值
    for key in ans_sub_exp.keys():
        ans_sub_exp[key] = np.divide(ans_sub_exp[key], ans_dist, \
                                out=np.zeros_like(ans_sub_exp[key]), where=ans_dist!=0)
    # print('Best combination calc time: {}s'.format(time.time()-start_time))
    return ScoredItem(ans_dist, ans_sub_exp)

def remove_worst_combination(item_list:list[ScoredItem]) -> ScoredItem:
    '''返回去除最差一件后的情况'''
    ans_item = ScoredItem([0], {})
    score_max = [len(item_list[i])-1 for i in range(len(item_list))]
    for i in range(len(item_list)):
        # 枚举最差位置
        for s in range(score_max[i]+1):
            c_dist = ScoredItem([1], {})
            # 枚举本件分数
            zero_mark = False
            for j in range(len(item_list)):
                if i == j:
                    continue
                # 容斥处理
                if j < i:
                    pos = s
                else:
                    pos = s+1
                if score_max[j] < pos:
                    # 最大值都没到枚举分数的情况肯定是不行的
                    zero_mark = True
                    break
                c_dist *= item_list[j][pos:]
            if zero_mark:
                continue
            ans_item += c_dist * item_list[i].score_dist.dist[s]
    return ans_item

def sim_select_best_k(item_set:list[Union[ScoredItem, np.ndarray]], k=2, sim_pairs=10000) -> np.ndarray:
    '''模拟选择k个最优'''
    if k > len(item_set):
        raise ValueError("k can't greater than item number")
    max_lenth = max([len(item)-1 for item in item_set]) * k + 1
    ans_dist = np.zeros(max_lenth, dtype=float)
    print("Begin simulate!")
    start_time = time.time()
    # 每个位置随机抽样
    sim_list = []
    for i, dist in enumerate(item_set):
        sim_result = np.random.choice(a=len(dist), size=sim_pairs, \
            p=dist if isinstance(dist, np.ndarray) else dist.score_dist.dist, replace=True)
        sim_list.append(sim_result)
    sim_array = np.column_stack(sim_list)
    sim_array.sort(axis=1)
    pos, value = np.unique(np.sum(sim_array[:, -k:], axis=1), return_counts=True)
    ans_dist[pos] = value
    print('Simulate select time: {}s'.format(time.time()-start_time))
    # 返回分布
    return ans_dist / sim_pairs

def check_subexp_sum(item: ScoredItem, weight: dict) -> np.ndarray:
    '''检查副词条的加权和'''
    ans = np.zeros(len(item))
    for key in item.sub_stats_exp.keys():
        ans[:len(item.sub_stats_exp[key])] += item.sub_stats_exp[key] * weight[key]
    return ans

# 下列这两个函数还没想好怎么封装比较好

def get_info(score_map: np.ndarray):
    '''
        获得分数差距条件下的主分布矩阵/副分布矩阵/对角概率/累计面积
        输入矩阵为方阵，0维度为主对象 1维度为副对象（其它选择）
    '''
    # 变量声明
    n = score_map.shape[0]
    main_dist = np.zeros_like(score_map)
    sub_dist = np.zeros_like(score_map)
    # 主对象大于副对象的情况
    main_dist[0, :] = np.sum(np.tril(score_map), axis=1)
    # 逐对角线计算
    for offset in range(1, n):
        main_dist[offset, :] = main_dist[offset-1, :]
        diag = score_map.diagonal(offset)
        main_dist[offset, 0:n-offset] += diag
        sub_dist[offset, offset:] = diag
    return main_dist, sub_dist

def get_mix_dist(listA: list[ScoredItem], listB: list[ScoredItem]):
    '''
        计算可从 listA 中选取一个部位替换为 listB 中对应位置情况下，最优决策下两者混合后的分布
    '''
    if len(listA) != len(listB):
        raise ValueError("A B lenth not equal!")
    # 计算最长边
    n = max([max(len(m[0]), len(m[1])) for m in zip(listA, listB)])
    m = len(listA)
    # for A, B in zip(listA, listB):
    #     n = max(n, len(A.score_dist), len(B.score_dist))
    # 计算对应的矩阵空间
    info_list = []
    for i, (A, B) in enumerate(zip(listA, listB)):
        M = np.outer(pad_zero(A.score_dist.dist, n), \
                     pad_zero(B.score_dist.dist, n))
        info_list.append(get_info(M))
    
    ans_item = ScoredItem([0], {})
    for i in range(m):
        # 枚举选择B的部位
        for s in range(1, n):
            # 枚举选择的分数差值，从1到n-1
            # 卷积条件分布，注意这里可能有值为0要特判
            if np.sum(info_list[i][1][s, :]) == 0:
                continue
            unit_item = ScoredItem(info_list[i][1][s, :], listB[i].sub_stats_exp)
            unit_item.sub_stats_clear()
            for j in range(m):
                # 遍历其他部位
                if i == j:
                    continue
                # 做容斥处理
                if i <= j:
                    s_pos = s
                else:
                    s_pos = s-1
                unit_item = unit_item * ScoredItem(FiniteDist(info_list[j][0][s_pos, :]), listA[j].sub_stats_exp)
            ans_item += unit_item
    
    # 添加不选择B的情况
    unit_item = ScoredItem([1], {})
    for i in range(m):
        unit_item = unit_item * ScoredItem(FiniteDist(info_list[i][0][0, :]), listA[i].sub_stats_exp)
    ans_item += unit_item
    return ans_item

if __name__ == '__main__':
    pass