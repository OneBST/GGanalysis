from GGanalysis.ScoredItem.scored_item import *
'''
    为词条评分类道具开发的组合评估工具
'''

__all__ = [
    'combine_items',
    'max_item',
    'max_item_set',
    'select_best_combination',
    'remove_worst_combination',
    'sim_select_best_k',
    'get_mix_dist',
]

def combine_items(item_list: list[ScoredItem]):
    '''返回列表内道具的混合'''
    ans = ScoredItem([1], {}, stats_score=item_list[0].stats_score)
    for item in item_list:
        ans *= item
    return ans

def max_item(a: ScoredItem, b: ScoredItem):
    '''
    返回两个道具混合后的最优分布，如果两者相等，副词条采用a而不是b
    '''
    cdf_a = dist2cdf(a.score_dist)
    cdf_b = dist2cdf(b.score_dist)
    min_len = min(len(a), len(b))
    max_len = max(len(a), len(b))
    ans_score = np.zeros(max_len, dtype=float)
    ans_sub_stats_exp = {}
    key_set = set(a.sub_stats_exp.keys())
    key_set.update(b.sub_stats_exp.keys())
    for key in key_set:
        ans_sub_stats_exp[key] = np.zeros(max_len, dtype=float)

    # a >= b 情况
    for i in range(min_len):
        p = cdf_b[i] * a.score_dist.dist[i]
        ans_score[i] += p
        for key in key_set:
            ans_sub_stats_exp[key][i] += a.sub_stats_exp.get(key, np.zeros(max_len, dtype=float))[i] * p
    # b > a 情况
    for i in range(1, min_len):
        p = cdf_a[i-1] * b.score_dist.dist[i]
        ans_score[i] += p
        for key in key_set:
            ans_sub_stats_exp[key][i] += b.sub_stats_exp.get(key, np.zeros(max_len, dtype=float))[i] * p
    # 副词条归一化
    for key in key_set:
        ans_sub_stats_exp[key] = np.divide(ans_sub_stats_exp[key], ans_score, out=np.zeros_like(ans_score), where=ans_score!=0)
    # 处理剩下部分
    if len(a) > len(b):
        c = a
    else:
        c = b
    ans_score[min_len:] = c.score_dist.dist[min_len:]
    for key in key_set:
        ans_sub_stats_exp[key][min_len:] = c.sub_stats_exp.get(key, np.zeros(max_len, dtype=float))[min_len:]
    return ScoredItem(score_dist=ans_score, sub_stats_exp=ans_sub_stats_exp, stats_score=a.stats_score)

def max_item_set(a: ScoredItemSet, b: ScoredItemSet):
    '''
    返回两个道具套装混合后的最优分布，如果两者分数相等，副词条采用a而不是b，如果套装部位不同，则采用有部位的
    '''
    item_set = {}
    key_set = set(a.item_set.keys())
    key_set.update(b.item_set.keys())
    for key in key_set:
        if key in a.item_set and key in b.item_set:
            item_set[key] = max_item(a.item_set[key], b.item_set[key])
        elif key in a.item_set:
            item_set[key] = a.item_set[key]
        elif key in b.item_set:
            item_set[key] = b.item_set[key]
    return ScoredItemSet(item_set)

class ConditionalScore():
    '''
    配合 select_best_combination 使用
    返回可行的分数序列（道具顺序）
    '''
    def __init__(self, item_idx:Union[list, tuple], score_max:list) -> None:
        self.state = [score_max[0]]
        self.item_idx = item_idx
        self.score_max = score_max
        # 生成分数初始状态
        for i in range(1, len(self.item_idx)):
            if self.item_idx[i] > self.item_idx[i-1]:
                # 规定一个方向可以取等，另一个方向不能取等以达到容斥效果
                self.state.append(min(self.score_max[i], self.state[i-1]))
            else:
                self.state.append(min(self.score_max[i], self.state[i-1])-1)
        # 定义合法分数序列集合
        self.possible_sequence = []
        # 枚举并选择合法分数序列，并排除枚举到末尾分数小于0的情况
        while self.state[-1] >= 0:
            self.possible_sequence.append(deepcopy(self.state))
            self.next_state()
        
    def next_state(self):
        '''切换到下一个可行状态，即找到按照不增规则的下一个'''
        pos = len(self.state)
        while pos >= 1:
            pos -= 1
            self.state[pos] -= 1
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
    
def select_best_combination(item_set:list[ScoredItem], chose_num=1) -> ScoredItem:
    '''
        返回选取最优chose_num件后的情况，chose_num<=2
        注意复杂度是关于以chose_num为幂次的多项式, 这个函数只在选择1或者选择2的情况下的容斥处理是对的
    '''
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
    return ScoredItem(ans_dist, ans_sub_exp, stats_score=item_set[0].stats_score)

def remove_worst_combination(item_list:list[ScoredItem]) -> ScoredItem:
    '''返回去除最差一件后的情况'''
    ans_item = ScoredItem([0], {}, stats_score=item_list[0].stats_score)
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

def get_mix_dist(listA: list[ScoredItem], listB: list[ScoredItem]) -> ScoredItem:
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
    
    ans_item = ScoredItem([0], {}, stats_score=listA[0].stats_score)
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