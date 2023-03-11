from matplotlib import font_manager
from matplotlib.font_manager import FontProperties  # 字体管理器
import matplotlib as mpl
import os.path as osp
import sys

__all__ = [
    'text_font',
    'title_font',
    'mark_font',
]

font_path = None
if sys.platform == 'win32':  # windows下
    font_path = 'C:/Windows/Fonts/'
else:  # Linux
    font_path = os.path.expanduser('~/.local/share/fonts/')

# 设置可能会使用的字体
font_w1 = font_manager.FontEntry(
    fname=osp.join(font_path,'SourceHanSansSC-Extralight.otf'),
    name='SHS-Extralight')
font_w2 = font_manager.FontEntry(
    fname=osp.join(font_path+'SourceHanSansSC-Light.otf'),
    name='SHS-Light')
font_w3 = font_manager.FontEntry(
    fname=osp.join(font_path,'SourceHanSansSC-Normal.otf'),
    name='SHS-Normal')
font_w4 = font_manager.FontEntry(
    fname=osp.join(font_path,'SourceHanSansSC-Regular.otf'),
    name='SHS-Regular')
font_w5 = font_manager.FontEntry(
    fname=osp.join(font_path,'SourceHanSansSC-Medium.otf'),
    name='SHS-Medium')
font_w6 = font_manager.FontEntry(
    fname=osp.join(font_path,'SourceHanSansSC-Bold.otf'),
    name='SHS-Bold')
font_w7 = font_manager.FontEntry(
    fname=osp.join(font_path,'SourceHanSansSC-Heavy.otf'),
    name='SHS-Heavy')

font_manager.fontManager.ttflist.extend([font_w1, font_w2, font_w3, font_w4, font_w5, font_w6, font_w7])

mpl.rcParams['font.sans-serif'] = [font_w4.name]

text_font = FontProperties('SHS-Medium', size=12)
title_font = FontProperties('SHS-Bold', size=18)
mark_font = FontProperties('SHS-Bold', size=12)