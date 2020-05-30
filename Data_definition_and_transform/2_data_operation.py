# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/5/30 15:33
filename(文件名): 2_data_operation.py
function description(功能描述):
...
    x, y, z为torch.tensor型数据，可以做各种操作或运算：
        加减乘除法转置(add, sub, mul, div, t)：
            z = x + y 或 z = torch.add(x, y)
            y.add_(x)  # inplace型加法

        像numpy一样的切片操作：
            x = y[:, 1]

        进行reshape
            x.view(new_shape) 如 x = torch.randn(4, 4), x.view(16)或x.view(-1, 8)
"""
import torch

x = torch.empty(3, 5)
print(x)
y = torch.ones_like(x)
y.add_(x)
print(y)
z = torch.t(y)
print(z)
