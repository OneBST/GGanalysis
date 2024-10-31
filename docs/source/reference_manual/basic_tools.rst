基础工具
======================================

随机变量分布处理工具
----------------------

.. autoclass:: GGanalysis.distribution_1d.FiniteDist
    :members:

.. automethod:: GGanalysis.distribution_1d.pad_zero

.. automethod:: GGanalysis.distribution_1d.cut_dist

.. automethod:: GGanalysis.distribution_1d.calc_expectation

.. automethod:: GGanalysis.distribution_1d.calc_variance

.. automethod:: GGanalysis.distribution_1d.dist2cdf

.. automethod:: GGanalysis.distribution_1d.cdf2dist

.. automethod:: GGanalysis.distribution_1d.linear_p_increase

.. automethod:: GGanalysis.distribution_1d.p2dist

.. automethod:: GGanalysis.distribution_1d.dist2p

.. automethod:: GGanalysis.distribution_1d.p2exp

.. automethod:: GGanalysis.distribution_1d.p2var


马尔可夫链处理工具
-------------------------

.. automethod:: GGanalysis.markov_method.table2matrix
    
.. automethod:: GGanalysis.markov_method.calc_stationary_distribution
    
.. automethod:: GGanalysis.markov_method.multi_item_rarity

.. autoclass:: GGanalysis.markov_method.PriorityPitySystem
    :members:

集齐问题处理工具
-------------------------

.. autoclass:: GGanalysis.coupon_collection.GeneralCouponCollection
    :members:

.. automethod:: GGanalysis.coupon_collection.get_equal_coupon_collection_exp
