'''
原神圣遗物数值表
''' 
ARTIFACT_TYPES = ['flower', 'plume', 'sands', 'goblet', 'circlet']

ARTIFACT_NAME = {
    'flower': '生之花', 
    'plume': '死之羽',
    'sands': '时之沙',
    'goblet': '空之杯',
    'circlet': '理之冠'
}

# 类型别名对照表
STAT_NAME = {
    'hp': '数值生命值',
    'hpp': '百分比生命值',
    'atk': '数值攻击力',
    'atkp': '百分比攻击力',
    'def': '数值防御力',
    'defp': '百分比防御力',
    'pyroDB': '火元素伤害加成',
    'hydroDB': '水元素伤害加成',
    'anemoDB': '风元素伤害加成',
    'electroDB': '雷元素伤害加成',
    'dendroDB': '草元素伤害加成',
    'cryoDB': '冰元素伤害加成',
    'geoDB': '岩元素伤害加成',
    'physicalDB': '物理伤害加成',
    'em': '元素精通',
    'er': '元素充能效率',
    'cr': '暴击率',
    'cd': '暴击伤害',
    'hb': '治疗加成',     
}

# 所有主词条掉落权重表
# 掉落权重取自 tesiacoil 的整理
# 主词条 https://wiki.biligame.com/ys/掉落系统学/常用数据#主词条
W_MAIN_STAT = {
    'flower': {'hp': 1000},
    'plume': {'atk': 1000},
    'sands': {
        'hpp': 1334,
        'atkp': 1333,
        'defp': 1333,
        'em': 500,
        'er': 500,
    },
    'goblet': {
        'hpp': 770,
        'atkp': 770,
        'defp': 760,
        'pyroDB': 200,
        'electroDB': 200,
        'cryoDB': 200,
        'hydroDB': 200,
        'dendroDB': 200,
        'anemoDB': 200,
        'geoDB': 200,
        'physicalDB': 200,
        'em': 100,
    },
    'circlet': {
        'hpp': 1100,
        'atkp': 1100,
        'defp': 1100,
        'cr': 500,
        'cd': 500,
        'hb': 500,
        'em': 200,
    }
}

# 所有副词条权重表
# 掉落权重取自 tesiacoil 的整理
# 副词条 https://wiki.biligame.com/ys/掉落系统学/常用数据#副词条
W_SUB_STAT = {
    'hp': 150,
    'atk': 150,
    'def': 150,
    'hpp': 100,
    'atkp': 100,
    'defp': 100,
    'em': 100,
    'er': 100,
    'cr': 75,
    'cd': 75,
}

# 五星圣遗物不同来源多词条占比 https://genshin-impact.fandom.com/wiki/Loot_System/Artifact_Drop_Distribution
P_DROP_STATE = {
    'domains_drop': 0.2,
    'normal_boos_drop': 0.34,
    'weekly_boss_drop': 0.34,
    'converted_by_alchemy_table': 0.34,
}

# 五星圣遗物主词条满级时属性 https://genshin-impact.fandom.com/wiki/Artifact/Stats
MAIN_STAT_MAX = {
    'hp': 4780,
    'atk': 311,
    'hpp': 0.466,
    'atkp': 0.466,
    'defp': 0.583,
    'em': 186.5,
    'er': 0.518,
    'pyroDB': 0.466,
    'hydroDB': 0.466,
    'anemoDB': 0.466,
    'electroDB': 0.466,
    'dendroDB': 0.466,
    'cryoDB': 0.466,
    'geoDB': 0.466,
    'physicalDB': 0.583,
    'cr': 31.1,
    'cd': 62.2,
    'hb': 35.9,
}

# 五星副词条单次升级最大值
# 全部的精确值见 https://nga.178.com/read.php?tid=31774495
SUB_STAT_MAX = {
    'hp': 298.75,
    'atk': 19.45,
    'def': 23.14,
    'hpp': 0.0583,
    'atkp': 0.0583,
    'defp': 0.0729,
    'em': 23.31 ,
    'er': 0.0648,
    'cr': 0.0389,
    'cd': 0.0777,
}

# 圣遗物强化经验翻倍概率 https://wiki.biligame.com/ys/掉落系统学/常用数据#主词条
P_EXP_MULTI = {
    '1': 0.9,
    '2': 0.09,
    '5': 0.01,
}

# 默认打分权重，这个值可以是 [0,1] 的浮点数
DEFAULT_STAT_SCORE = {
    'hp': 0,
    'atk': 0,
    'def': 0,
    'hpp': 0,
    'atkp': 1,
    'defp': 0,
    'em': 0,
    'er': 0,
    'cr': 1,
    'cd': 1,
}

# 默认主属性选择
DEFAULT_MAIN_STAT = {
    'flower': 'hp',
    'plume': 'atk',
    'sands': 'atkp',
    'goblet': 'pyroDB',
    'circlet': 'cr',
}

# 默认颜色
DEFAULT_STAT_COLOR = {
    'hp': '#4e8046',
    'hpp': '#65a65b',
    'atk': '#8c4646',
    'atkp': '#b35959',
    'def': '#8c8c3f',
    'defp': '#b2b350',
    'pyroDB': '#d96857',
    'hydroDB': '#6c77d9',
    'anemoDB': '#4dbf99',
    'electroDB': '#c566cc',
    'dendroDB': '#59b364',
    'cryoDB': '#6cd5d9',
    'geoDB': '#bfb560',
    'physicalDB': '#7a8c99',
    'em': '#a15ba6',
    'er': '#665ba6',
    'cr': '#f29224',
    'cd': '#f24124',
    'hb': '#79a63a',     
}