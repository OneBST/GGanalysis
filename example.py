import GGanalysisLite as ggl

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