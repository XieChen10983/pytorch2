# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/5/30 12:30
filename(�ļ���): test.py
function description(��������):
...
"""
import torch

a = torch.randn(5, 6)
b = torch.randn(6, 7)
print(a.mm(b))
