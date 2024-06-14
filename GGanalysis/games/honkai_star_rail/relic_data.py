'''
    崩铁遗器数值表
''' 
RELIC_TYPES = ['head', 'hands', 'body', 'feet', 'link rope', 'planar sphere']

RELIC_SETS = {
    'cavern relics': '遗器套装',
    'planar ornaments': '位面饰品',
}

CAVERN_RELICS = ['head', 'hands', 'body', 'feet']
PLANAR_ORNAMENTS = ['link rope', 'planar sphere']

RELIC_NAME = {
    'head': '头部', 
    'hands': '手部',
    'body': '躯干',
    'feet': '脚部',
    'planar sphere': '位面球',
    'link rope': '连结绳',
}

# 类型别名对照表
STAT_NAME = {
    'hp': '数值生命值',
    'atk': '数值攻击力',
    'def': '数值防御力',
    'hpp': '百分比生命值',
    'atkp': '百分比攻击力',
    'defp': '百分比防御力',
    'speed': '速度',
    'physicalDB': '物理伤害加成',
    'fireDB': '火属性伤害加成',
    'iceDB': '冰属性伤害加成',
    'lightningDB': '雷属性伤害加成',
    'windDB': '风属性伤害加成',
    'quantumDB': '量子属性伤害加成',
    'imaginaryDB': '虚数属性伤害加成',
    'cr': '暴击率',
    'cd': '暴击伤害',
    'ehr': '效果命中',
    'er': '效果抵抗',
    'be': '击破特攻',
    'err': '能量恢复效率',
    'hb': '治疗量加成',
}

# 所有主词条掉落权重表
W_MAIN_STAT = {
    'head': {'hp': 1000},
    'hands': {'atk': 1000},
    'body': {
        'hpp': 1000,
        'atkp': 1000,
        'defp': 1000,
        'cr': 500,
        'cd': 500,
        'hb': 500,
        'ehr': 500,
    },
    'feet': {
        'hpp': 1375,
        'atkp': 1500,
        'defp': 1500,
        'speed': 625,
    },
    'link rope': {
        'hpp': 1375,
        'atkp': 1375,
        'defp': 1225,
        'be': 750,
        'err': 275,
    },
    'planar sphere': {
        'hpp': 625,
        'atkp': 625,
        'defp': 600,
        'physicalDB': 450,
        'fireDB': 450,
        'iceDB': 450,
        'lightningDB': 450,
        'windDB': 450,
        'quantumDB': 450,
        'imaginaryDB': 450,
    },
}

# 所有副词条权重表
W_SUB_STAT = {
    'hp': 100,
    'atk': 100,
    'def': 100,
    'hpp': 100,
    'atkp': 100,
    'defp': 100,
    'speed': 40,
    'cr': 60,
    'cd': 60,
    'ehr': 80,
    'er': 80,
    'be': 80,
}

# 五星遗器初始4词条概率
P_INIT4_DROP = 0.2

# 五星遗器主词条满级时属性 https://www.bilibili.com/video/BV11w411U7AV/
MAIN_STAT_MAX = {
    'hp': 705.6,
    'atk': 352.8,
    'hpp': 0.432,
    'atkp': 0.432,
    'defp': 0.54,
    'cr': 0.324,
    'cd': 0.648,
    'hb': 0.345606,
    'ehr': 0.432,
    'speed': 25.032,
    'be': 0.648,
    'err': 0.194394,
    'physicalDB': 0.388803,
    'fireDB': 0.388803,
    'iceDB': 0.388803,
    'lightningDB': 0.388803,
    'windDB': 0.388803,
    'quantumDB': 0.388803,
    'imaginaryDB': 0.388803,
}

# 五星副词条单次升级最大值，升级挡位为三挡比例 8:9:10 https://honkai-star-rail.fandom.com/wiki/Relic/Stats
SUB_STAT_MAX = {
    'hp': 42.338,
    'atk': 21.169,
    'def': 21.169,
    'hpp': 0.0432,
    'atkp': 0.0432,
    'defp': 0.054,
    'speed': 2.6,
    'cr': 0.0324,
    'cd': 0.0648,
    'ehr': 0.0432,
    'er': 0.0432,
    'be': 0.0648,
}

# 默认打分权重，这个值可以是 [0,1] 的浮点数
DEFAULT_STAT_SCORE = {
    'hp': 0,
    'atk': 0,
    'def': 0,
    'hpp': 0,
    'atkp': 0.5,
    'defp': 0,
    'speed': 1,
    'cr': 1,
    'cd': 1,
    'ehr': 0,
    'er': 0,
    'be': 0,
}

# 默认主词条得分
DEFAULT_MAIN_STAT_SCORE = {
    'hp': 0,
    'atk': 0,
    'hpp': 0,
    'atkp': 0,
    'defp': 0,
    'cr': 0,
    'cd': 0,
    'hb': 0,
    'ehr': 0,
    'speed': 0,
    'be': 0,
    'err': 0,
    'physicalDB': 0,
    'fireDB': 0,
    'iceDB': 0,
    'lightningDB': 0,
    'windDB': 0,
    'quantumDB': 0,
    'imaginaryDB': 0,
}

# 默认主属性选择
DEFAULT_MAIN_STAT = {
    'head': 'hp',
    'hands': 'atk',
    'body': 'cr',
    'feet': 'speed',
    'planar sphere': 'physicalDB',
    'link rope': 'atkp',
}

# 默认颜色
DEFAULT_STAT_COLOR = {
    'hp': '#4e8046',
    'atk': '#8c4646',
    'def': '#8c8c3f',
    'hpp': '#65a65b',
    'atkp': '#b35959',
    'defp': '#b2b350',
    'speed': '#a15ba6',
    'physicalDB': '#7a8c99',
    'fireDB': '#d96857',
    'iceDB': '#6cb1d9',
    'lightningDB': '#c566cc',
    'windDB': '#4dbf99',
    'quantumDB': '#712a79',
    'imaginaryDB': '#edd92d',
    'cr': '#f29224',
    'cd': '#f24124',
    'ehr': '#617349',
    'er': '#563aa6',
    'be': '#13eaed',
    'err': '#665ba6',
    'hb': '#79a63a', 
}