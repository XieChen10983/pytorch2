# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/5/30 15:05
filename(文件名): 1_data_definition.py
function description(功能描述):
...
    torch生成以下类型的张量：
        1. 空张量： torch.empty(5, 3)
        2. 全0张量: torch.zeros(5, 3)
        3. 全1张量: torch.ones(5, 3)
        4. 随机张量: torch.rand(5, 3)
    可在后面加dtype，如torch.zeros(5, 3, dtype=torch.long)，torch的数据类型有：
        torch.long
        torch.float
        torch.int
        torch.double
        torch.int64
        torch.uint8...
    获取张量x的size：
        x.size(), x.size(0)等
    获取张量x的dtype：
        x.dtype

    获取有梯度的张量：requires_grad=True 或 x.requires_grad_(True), 只有梯度有张量，才能进行back-prop
        x = torch.Tensor([3, 5], requires_grad=True)
    由有梯度的变量生成的变量也是有梯度的。
        y = x*x  =>  y.requires_grad = True

    当有梯度的变量进行计算时，与该变量相关的所有变量梯度都会发生变化，为了避免这个发生：
        对于单个变量x：x.detach() 或 x.requires_grad_(False)
        对于一整段代码中的变量: with torch.no_grad():

    张量梯度的计算：
        scalar.backward()
    张量梯度的获取：
        x.grad
"""
import torch

x = torch.empty((5, 3), dtype=torch.uint8)
print(x.requires_grad)
y = x*x
print(y.requires_grad)

x = torch.tensor([5.5, 3], requires_grad=True)
z = torch.tensor([7, 6], requires_grad=False)
print(x.requires_grad)
y = x*z
y = torch.sum(y)
print(y.requires_grad)

with torch.no_grad():
    y.backward()
    print(x.grad)

print(x.size())
print(x.dtype)
