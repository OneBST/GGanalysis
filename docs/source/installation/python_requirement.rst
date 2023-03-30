安装 Python 及依赖包
========================

安装 Python
------------------------

工具包需要 ``python>=3.8``，可以从 `官方网站 <https://www.python.org/>`_ 下载并安装。

也可以将使用环境置于 Anaconda 环境中， `Anaconda下载 <https://www.anaconda.com//>`_ `Miniconda下载 <https://docs.conda.io/en/latest/miniconda.html/>`_。

.. attention:: 

   只使用抽卡概率计算相关工具时可以采用更老的 Python 版本（不推荐）。
   Python 3.8 之前的版本不支持类型提示，在导入圣遗物计算相关包时会报错，可以选择手动修改报错部分类型提示代码解决问题。

安装依赖包
------------------------

GGanalysis 依赖的 Python 包如下：

   - `numpy <https://numpy.org/>`_
   - `scipy <https://scipy.org/>`_
   - `matplotlib <https://matplotlib.org/>`_

可以选择参考以下代码手动安装依赖包，或者直接参考 :ref:`安装 GGanalysis <install_gganalysis>` 节中安装 GGanalysis 包的过程，一切正常的话依赖包会自动安装。

.. code:: shell

   pip install numpy scipy matplotlib