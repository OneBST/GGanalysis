配置环境并安装工具包
========================

安装 Python
------------------------

工具包需要 ``python>=3.9``，可以从 `官方网站 <https://www.python.org/>`_ 下载并安装。

也可以将使用环境置于 Anaconda 环境中， `Anaconda下载 <https://www.anaconda.com//>`_ `Miniconda下载 <https://docs.conda.io/en/latest/miniconda.html/>`_。

.. attention:: 

   只使用抽卡概率计算相关工具时可以采用更老的 Python 版本（不推荐）。
   Python 3.9 之前的版本不支持工具包内类型提示写法，在导入计算相关包时会报错，可以选择手动修改报错部分类型提示代码解决问题。

安装依赖包
------------------------

GGanalysis 依赖的 Python 包如下：

   - `numpy <https://numpy.org/>`_
   - `scipy <https://scipy.org/>`_
   - `matplotlib <https://matplotlib.org/>`_

可以选择参考以下代码手动安装依赖包，或者直接参考下节中安装 GGanalysis 包的过程，一切正常的话依赖包会自动安装。

.. code:: shell

   pip install numpy scipy matplotlib

安装 GGanalysis
------------------------

在本地机器安装了 ``git`` 的情况下，可以直接使用 ``git clone`` 命令，并使用 ``pip`` 安装。

.. code:: shell

    git clone https://github.com/OneBST/GGanalysis.git
    cd GGanalysis
    # 如果不需要编辑工具包内代码，直接执行以下命令，安装完成后 git 下载文件可删除
    pip install .
    # 如果需要编辑工具包内代码，加入 -e 选项，运行时直接引用当前位置的包，对代码的修改会实时反应
    pip install -e .

如果没有安装 ``git`` ，可以点击 `这个链接 <https://github.com/OneBST/GGanalysis/archive/refs/heads/main.zip>`_ 下载压缩包。
解压后按照以上流程操作。（注意需要将 ``cd GGanalysis`` 改为 ``cd GGanalysis-main``）

安装画图使用字体
------------------------

.. attention:: 

   安装画图程序所需字体是不是必须的。如果只需要使用 GGanalysis 工具包进行概率计算，则可以略过此步骤。

GGanalysis 工具包使用 `思源黑体 <https://github.com/adobe-fonts/source-han-sans>`_ ，
请下载工具包使用的 `特定版本字体 <https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip>`_ 并安装。