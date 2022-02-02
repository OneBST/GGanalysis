import GGanalysisLite as ggl

# “非酋度”及“欧皇度”计算
permanent_5 = 5			# 常驻祈愿五星数量
permanent_pull = 301	# 常驻祈愿恰好出最后一个五星时花费的抽数
character_5 = 14		# 角色祈愿五星数量
character_pull = 876	# 角色祈愿恰好出最后一个五星时花费的抽数
character_up_5 = 8		# 角色祈愿UP五星数量
character_up_pull = 876	# 角色祈愿恰好出最后一个UP五星时花费的抽数
weapon_5 = 2			# 武器祈愿五星数量
weapon_pull = 126		# 武器祈愿恰好出最后一个五星时花费的抽数

# 初始化玩家
player = ggl.GenshinPlayer(	p5=permanent_5,
                            c5=character_5,
                            u5=character_up_5,
                            w5=weapon_5,
                            p_pull=permanent_pull,
                            c_pull=character_pull,
                            u_pull=character_up_pull,
                            w_pull=weapon_pull)

# 查看常驻祈愿rank
print('常驻祈愿', player.get_p5_rank())

# 查看角色祈愿rank（考虑五星数量）
print('角色祈愿', player.get_c5_rank())

# 查看UP角色rank（考虑UP五星数量）
print('角色祈愿UP', player.get_u5_rank())

# 查看武器祈愿rank
print('武器祈愿', player.get_w5_rank())

# 查看综合rank（角色祈愿考虑五星数量）
print('综合', player.get_comprehensive_rank())

# 查看综合rank（角色祈愿考虑UP数量）
print('综合UP', player.get_comprehensive_rank_Up5Character())


# 条件分布列计算
# 计算垫了30抽，有大保底，抽取2个UP五星角色所需抽数的分布列
c = ggl.Up5starCharacter()
dist_c = c.conditional_distribution(2, pull_state=30, up_guarantee=1)

# 计算垫了10抽，有大保底，命定值为0，抽取2个定轨UP五星武器所需抽数的分布列
w = ggl.UP5starWeaponEP()
dist_w = w.conditional_distribution(2, pull_state=10, up_guarantee=1, fate_point=0)

# 计算在没有垫抽情况下获得2个UP角色和1个定轨五星武器的抽数分布列
dist_cw = ggl.GI_conv_cw(3, 0, 0, 1, 0, 0, 0)