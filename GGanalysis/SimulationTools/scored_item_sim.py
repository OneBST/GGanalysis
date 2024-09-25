import numpy as np
import random
from copy import deepcopy
from GGanalysis.SimulationTools.statistical_tools import Statistics
import multiprocessing
from tqdm import tqdm

class HoyoItemSim():
    def __init__(
            self,
            stat_score,     # 副词条评分
            item_type,      # 道具类型
            main_stat,      # 设定主词条
            W_MAIN_STAT,    # 主词条权重
            W_SUB_STAT,     # 副词条权重
            UPDATE_TIMES,   # 后续可升级次数分布
            UPDATE_TICKS,   # 词条属性数值分布
            ) -> None:
        self.stat_score = stat_score

        self.type = item_type
        self.main_stat = main_stat
        self.main_stat_weight = W_MAIN_STAT
        self.sub_stats_weight = deepcopy(W_SUB_STAT)
        if self.main_stat in self.sub_stats_weight:
            del self.sub_stats_weight[self.main_stat]
        # 从字典中提取键和权重
        self.sub_stat_keys = list(self.sub_stats_weight.keys())
        # 归一化权重
        self.sub_stat_weights = list(self.sub_stats_weight.values())
        total_weight = sum(self.sub_stat_weights)
        self.sub_stat_weights = [w / total_weight for w in self.sub_stat_weights]

        self.UPDATE_TIMES = UPDATE_TIMES
        self.UPDATE_TICKS = UPDATE_TICKS
    
    def sample(self):
        # 是否满足主词条
        if random.random() > self.main_stat_weight[self.type][self.main_stat] / sum(self.main_stat_weight[self.type].values()):
            return 0, {}
        # 选择副词条升级次数
        update_times = np.random.choice(range(len(self.UPDATE_TIMES)), p=self.UPDATE_TIMES)
        # 选择副词条
        selected_stats = np.random.choice(self.sub_stat_keys, size=4, replace=False, p=self.sub_stat_weights)
        # 确定副词条初始数值
        stat_value = {}
        for stat in selected_stats:
            stat_value[stat] = np.random.choice(range(len(self.UPDATE_TICKS)), p=self.UPDATE_TICKS)
        # 升级副词条数值（均匀分布）
        for i in range(update_times):
            stat = selected_stats[random.randint(0, 3)]
            stat_value[stat] += np.random.choice(range(len(self.UPDATE_TICKS)), p=self.UPDATE_TICKS)
        # 计算部位得分
        score = 0
        for stat in selected_stats:
            score += self.stat_score[stat] * stat_value[stat]
        return score, stat_value

    def sample_best(self, n):
        # 重复 n 次后的最好值
        best_score = 0
        best_stat_value = {}
        for i in range(n):
            score, stat_value = self.sample()
            if score > best_score:
                best_score = score
                best_stat_value = stat_value
        return best_score, best_stat_value

class HoyoItemSetSim():
    def __init__(self, items, items_p, set_p, stat_score) -> None:
        self.items:list[HoyoItemSim] = items
        self.items_p:list[float] = items_p
        self.set_p = set_p
        self.stat_score = stat_score

    def sample_best(self, n):
        set_n = np.random.binomial(n, self.set_p)
        item_nums = np.random.multinomial(set_n, self.items_p)
        score = 0
        stat_value = {}
        for num, item in zip(item_nums, self.items):
            item_score, item_stat = item.sample_best(num)    
            score += item_score
            for key in item_stat.keys():
                if key in stat_value:
                    stat_value[key] += item_stat[key]
                else:
                    stat_value[key] = item_stat[key]
        return score, stat_value

    def sim_player_group(self, player_num, n, record_sub_stat_dist=False):
        # 总共 player_num 个玩家，每个玩家获得 n 件道具
        score_record = Statistics(is_record_dist=True)
        stat_record = {}

        def record_sub_stat(stat_value):
            for key in self.stat_score.keys():
                if key not in stat_record:
                    stat_record[key] = Statistics(record_sub_stat_dist)
                stat_record[key].update(stat_value.get(key, 0))
        for _ in range(player_num):
            score, stat_value = self.sample_best(n)
            score_record.update(score)
            record_sub_stat(stat_value)
        
        return score_record, stat_record

    def parallel_sim_player_group(self, player_num, n, forced_processes=None, record_sub_stat_dist=False):
        # 总共 player_num 个玩家，每个玩家获得 n 件道具
        num_processes = multiprocessing.cpu_count()
        if forced_processes is not None:
            num_processes = forced_processes

        with multiprocessing.Pool(num_processes) as pool:
            # 使用map方法将计算任务分发给所有进程
            results = list(tqdm(pool.imap(self.sample_best, [n]*player_num), total=player_num))

        score_record = Statistics(is_record_dist=True)
        stat_record = {}
        def record_sub_stat(stat_value):
            for key in self.stat_score.keys():
                if key not in stat_record:
                    stat_record[key] = Statistics(record_sub_stat_dist)
                stat_record[key].update(stat_value.get(key, 0))
        
        for score, stat_value in results:
            score_record.update(score)
            record_sub_stat(stat_value)
        # 返回处理后的结果
        return score_record, stat_record

if __name__ == '__main__':
    from GGanalysis.games.honkai_star_rail.relic_data import W_MAIN_STAT,W_SUB_STAT,CAVERN_RELICS,PLANAR_ORNAMENTS,DEFAULT_MAIN_STAT,DEFAULT_STAT_SCORE
    test_item = HoyoItemSim(
        DEFAULT_STAT_SCORE,
        CAVERN_RELICS[0],
        list(W_MAIN_STAT[CAVERN_RELICS[0]].keys())[0],
        W_MAIN_STAT,
        W_SUB_STAT,
        [0,0,0,0.8,0.2],
        [0,0,0,0,0,0,0,0,1/3,1/3,1/3],
    )
    # print(test_item.sample())
    # print(test_item.sample_best(1000))
    
    test_item_set = HoyoItemSetSim([test_item], [1], 1/2, DEFAULT_STAT_SCORE)
    # print(test_item_set.sample_best(1000))
    import time
    t0 = time.time()
    print(test_item_set.sim_player_group(100, 1000))
    t1 = time.time()
    print(test_item_set.parallel_sim_player_group(100, 1000, 16))
    t2 = time.time()
    print(t1-t0, t2-t1, "Rate =", (t1-t0)/(t2-t1))