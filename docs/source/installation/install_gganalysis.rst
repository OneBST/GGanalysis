.. _install_gganalysis:

安装 GGanalysis
========================

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


