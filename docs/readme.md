# GGanalysis 工具包文档

文档源文件在 `docs/source/`，使用 Sphinx、Furo 和 reStructuredText。

### 配置方法

``` shell
python -m pip install -e .
python -m pip install -r docs/requirements.txt
```

上述命令从仓库根目录执行。autodoc 会导入项目代码，因此需要安装项目依赖，
包括 `matplotlib`；缺少依赖时即使生成 HTML，API 页面也可能不完整。

### 构建与检查

从仓库根目录执行：

``` shell
python -m sphinx -M html docs/source ../ggdoc -W --keep-going
```

Windows 也可在 `docs/` 中执行 `.\make.bat html`，其他平台可执行 `make html`。
输出统一为同级 `ggdoc/html/` 和 `ggdoc/doctrees/`，不放入源码目录。
本机网页目录为 `C:\Users\Mu\Documents\FileCan\PersonalProjects\ggdoc\html`。

新增页面须接入 `toctree`。检查构建警告、首页和侧边栏入口、交叉引用、API 签名，
并运行新增或修改的 Python 示例。临时验证脚本及报告放在被忽略的 `test/`。
核心定义入口见 `docs/source/reference_manual/index.rst`，开发约定见根目录 `AGENTS.md`。

