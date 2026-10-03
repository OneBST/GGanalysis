# GGanalysis 开发约定

## 先查阅，再实现
- 模型基类及现成模型：`GGanalysis/basic_models.py`（`GachaModel`、
  `CommonGachaModel`、`PityModel`、`DualPityModel` 等）。
- 抽卡层定义与组合：`GGanalysis/gacha_layers.py`（`GachaLayer`、
  `PityLayer`、`BernoulliLayer`、`MarkovLayer` 等）。
- 一维分布与卷积：`GGanalysis/distribution_1d.py`（`FiniteDist`）；
  跨阶段状态与花费：`GGanalysis/state_distribution.py`（`StateDist`、`StateKernel`）。
- 状态转移、命中和稳态分析：`GGanalysis/markov/`；装备评分：
  `GGanalysis/scored_item/`；绘图：`GGanalysis/gacha_plot.py`、`GGanalysis/plot_tools.py`。
- 模型开发或新增游戏前，阅读 [开发指南](docs/source/development.rst) 的相关部分
  及机制相近的已有实现。优先复用；新增公共工具前先检索已有能力。

## 模型与代码
- 明确目标事件、花费单位、初始状态及重置/继承规则。区分首次条件分布、
  后续 IID 周期和跨次状态依赖；普通卷积要求花费独立，同一分布的卷积幂要求 IID。
- 沿用已有分布类型、索引和转移方向约定。规则常量单处定义，计算、绘图和模拟
  引用同一来源；通用数学能力放公共模块，游戏特有规则留在游戏目录，避免过度抽象。
- 新游戏最低提供 `gacha_model.py` 模型入口、`__init__.py` 公开导出及明确的
  `__all__`、机制来源/假设和已接入目录的最小游戏文档；其他组件按需增加。
- 语义适用时沿用 `item_num`、`multi_dist`、`item_pity` 等参数和返回约定。
  新增代码使用明确导入、类型注解和中文说明，保持局部格式一致；
  不顺带重排无关代码，不擅自改变公开接口或覆盖已有工作区修改。

## 文档与验证
- 正式文档用中文 RST，放在 `docs/source/` 并接入 `toctree`；API 细节维护在
  docstring，通过 autodoc 引用。新增公开接口写清参数、返回值和适用假设。
  接口或机制变更同步更新文档，文档变更检查 Sphinx 构建；从根目录执行
  `python -m sphinx -M html docs/source ../ggdoc`，网页输出到同级 `ggdoc/html/`。
- 测试、基准测试和临时验证报告统一放在被忽略的 `test/`，暂不进入 Git，
  不强制添加；可复用模拟器属于正式源码。重要数学假设和验证结论保留在文档中。
- 默认使用解析、递推或状态方法，按改动检查概率质量、边界和已知结果。
  模拟仅用于必要的独立交叉验证，或确定性计算不可行的情况；记录种子、样本量和误差。
- 不用强制归一化掩盖概率丢失；明确截断与近似。完成后说明改动、实际验证结果和限制。
