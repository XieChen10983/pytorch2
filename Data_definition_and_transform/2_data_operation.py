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

        二维矩阵乘法(mm) 或 @：
            x.mm(y) 或 x @ y
        三维矩阵乘法(第一维是batchsize)：torch.bmm(x, y)
        混合矩阵乘法(较为复杂，具体查看pytorch教程)：torch.matmul(x, y)
        逐点乘法：torch.mul(x, y)

        像numpy一样的切片操作：
            x = y[:, 1]

        进行reshape
            x.view(new_shape) 如 x = torch.randn(4, 4), x.view(16)或x.view(-1, 8)
"""
import torch

x = torch.empty(3, 5)
print(x)
y = torch.ones_like(x)
# 自加法
y.add_(x)
print(y)

# 矩阵乘法
x1 = torch.randint(1, 4, (3, 5))
x2 = torch.randint(1, 4, (5, 3))
output = x1.mm(x2)
print(output.size())
output2 = x1 @ x2
print(output2.size())

# 转置
z = torch.t(y)
print(z)
