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
"""
import torch

x = torch.empty((5, 3), dtype=torch.uint8)
print(x)

# x = torch.tensor([5.5, 3], dtype=torch.float)
# print(x)

print(x.size())
print(x.dtype)
