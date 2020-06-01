# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/5/30 12:30
filename(文件名): test.py
function description(功能描述):
...
"""
import torch

a = torch.randn(5, 6)
b = torch.randn(6, 7)
print(a.mm(b))
