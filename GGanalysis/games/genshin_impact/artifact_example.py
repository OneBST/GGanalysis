import GGanalysis as gg
import GGanalysis.games.genshin_impact as GI
from GGanalysis.ScoredItem import combine_items

# 注意，以下定义的分数指每次副词条强化为最高属性为10，其他情况依次为9、8、7

# 定义圣遗物生之花
flower = GI.GenshinArtifact(type='flower')
print('在获取100件生之花后，获取的最高分数分布')
print(flower.repeat(100, p=1).score_dist.__dist)
print('在祝圣秘境获取100件圣遗物后，其中特定套装的生之花的最高分数的分布')
print(flower.repeat(100).score_dist.__dist)

# 定义以默认权重计算的圣遗物套装
default_weight_artifact = GI.GenshinArtifactSet()
print('在祝圣秘境获取100件圣遗物后，其中特定套装组5件套能获得的最高分数的分布')
print(combine_items(default_weight_artifact.repeat(100)).score_dist.__dist)
print('在祝圣秘境获取100件圣遗物，获取散件1500件后，其中特定套装和散件组4+1能获得的最高分数的分布')
print(default_weight_artifact.get_4piece_under_condition(n=100, base_n=1500).score_dist.__dist)

# 自定义属性，原神词条别名说明见 games/genshin_impact/artifact_data.py
# 自定义主词条选择
MAIN_STAT = {
    'flower': 'hp',
    'plume': 'atk',
    'sands': 'atkp',
    'goblet': 'pyroDB',
    'circlet': 'cd',
}
# 自定义副词条权重
STAT_SCORE = {
            'atkp': 0.5,
            'em': 0.6,
            'er': 0.3,
            'cr': 1,
            'cd': 1,
}
custom_weight_artifact = GI.GenshinArtifactSet(main_stat=MAIN_STAT, stats_score=STAT_SCORE)
print('自定义条件下，在祝圣秘境获取100件圣遗物，获取散件1500件后，其中特定套装和散件组4+1能获得的最高分数的分布')
print(default_weight_artifact.get_4piece_under_condition(n=100, base_n=1500).score_dist.__dist)
