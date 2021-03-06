# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/5/30 15:45
filename(文件名): 3_data_transformation.py
function description(功能描述):
...
    转换为python数据
        如果tensor x只有单个元素，那么可以用x.item()得到python型数据
    转换为numpy数据
        y = x.numpy()
        此时数据之间是联动的，即若x发生变化，则y随之发生变化。
    从numpy数据转为tensor
        z = np.ones(3, 5)
        x = torch.from_numpy(z)
        此时数据之间也是联动的，即若numpy数据z发生变化，tensor数据x也随之变化
    将数据转换为cuda数据(GPU计算时，数据和网络都需要转为GPU类型，即data.to(device), net.to(device))：
        判断是否有cuda：
            torch.cuda.is_available()
        cpu数据转为cuda数据：
            device = torch.device("cuda")
            x = torch.ones(3, 5, device=device)或
            x = torch.ones(3, 5), x = x.to(device)
"""
import torch
import numpy as np

x = torch.randn(1, requires_grad=True)
print(x)
print(x.item())
print(x.detach().numpy())

y = np.ones((3, 5))
z = torch.mul(x, x)
np.add(y, 1, out=y)
print(z)

print(x.requires_grad)
# print(y.requires_grad)
print(z.requires_grad)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(3, 5, device=device)
    print(x)
