# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/5/30 15:05
filename(�ļ���): 1_data_definition.py
function description(��������):
...
    torch�����������͵�������
        1. �������� torch.empty(5, 3)
        2. ȫ0����: torch.zeros(5, 3)
        3. ȫ1����: torch.ones(5, 3)
        4. �������: torch.rand(5, 3)
    ���ں����dtype����torch.zeros(5, 3, dtype=torch.long)��torch�����������У�
        torch.long
        torch.float
        torch.int
        torch.double
        torch.int64
        torch.uint8...
    ��ȡ����x��size��
        x.size(), x.size(0)��
    ��ȡ����x��dtype��
        x.dtype
"""
import torch

x = torch.empty((5, 3), dtype=torch.uint8)
print(x)

# x = torch.tensor([5.5, 3], dtype=torch.float)
# print(x)

print(x.size())
print(x.dtype)
