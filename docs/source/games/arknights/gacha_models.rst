.. _arknights_gacha_model:

明日方舟抽卡模型
========================

**GGanalysis** 使用基本的抽卡模板模型结合 `明日方舟抽卡系统参数 <https://www.bilibili.com/read/cv20251111>`_ 定义了一系列可以直接取用的抽卡模型。需要注意的是明日方舟的抽卡系统模型的确定程度并没有很高，使用时需要注意。

此外，还针对性编写了如下模板模型：

    适用于计算 `定向寻访 <https://www.bilibili.com/read/cv22596510>`_ 时获得特定UP六星角色的模型
    :class:`~GGanalysis.games.arknights.AKDirectionalModel`

    适用于计算通过 `统计数据 <https://www.bilibili.com/video/BV1ib411f7YF/>`_ 发现的类型硬保底的模型
    :class:`~GGanalysis.games.arknights.AKHardPityModel`

    适用于计算集齐多种六星的模型（不考虑300井、定向寻访及类型硬保底机制）
    :class:`~GGanalysis.games.arknights.AK_Limit_Model` 