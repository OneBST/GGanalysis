from GGanalysis.SimulationTools.scored_item_sim import HoyoItemSim, HoyoItemSetSim
from GGanalysis.games.honkai_star_rail.relic_data import W_MAIN_STAT,W_SUB_STAT,CAVERN_RELICS,PLANAR_ORNAMENTS,DEFAULT_MAIN_STAT,DEFAULT_STAT_SCORE

class StarRailRelicSim(HoyoItemSim):
    # 崩铁遗器单件
    def __init__(self, stat_score, item_type, main_stat) -> None:
        super().__init__(
            stat_score,
            item_type,
            main_stat,
            W_MAIN_STAT,
            W_SUB_STAT,
            [0,0,0,0,0.8,0.2],
            [0,0,0,0,0,0,0,0,1/3,1/3,1/3]
        )

class StarRailRelicCavernRelicsSim(HoyoItemSetSim):
    # 崩铁遗器四件套
    def __init__(self, main_stats, stat_score) -> None:
        items = []
        items_p = [1/(len(CAVERN_RELICS))] * len(CAVERN_RELICS)
        for item_type in CAVERN_RELICS:
            items.append(StarRailRelicSim(stat_score, item_type, main_stats[item_type]))
        super().__init__(items, items_p, 1/2, stat_score)

class StarRailRelicPlanarOrnamentsSim(HoyoItemSetSim):
    # 崩铁遗器两件套
    def __init__(self, main_stats, stat_score) -> None:
        items = []
        items_p = [1/(len(PLANAR_ORNAMENTS))] * len(PLANAR_ORNAMENTS)
        for item_type in PLANAR_ORNAMENTS:
            items.append(StarRailRelicSim(stat_score, item_type, main_stats[item_type]))
        super().__init__(items, items_p, 1/2, stat_score)

if __name__ == '__main__':
    # Windows 下 Python 多线程必须使用 "if __name__ == '__main__':"
    sim_players = 10000
    sim_4set = StarRailRelicCavernRelicsSim(DEFAULT_MAIN_STAT, DEFAULT_STAT_SCORE)
    score, sub_stat = sim_4set.parallel_sim_player_group(sim_players, 2000)
    # print(sim_4set.sample_best(2000))
    print("Sim Players:", sim_players)
    print("Total Score:", score.mean/9)
    print({key:sub_stat[key].mean/9 for key in ['atkp', 'cr', 'cd', 'speed']})
    print('Sub stat score sum', sum([sub_stat[key].mean/9*DEFAULT_STAT_SCORE[key] for key in DEFAULT_STAT_SCORE.keys()]))