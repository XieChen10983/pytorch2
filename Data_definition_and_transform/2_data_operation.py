# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/5/30 15:33
filename(�ļ���): 2_data_operation.py
function description(��������):
...
    x, y, zΪtorch.tensor�����ݣ����������ֲ��������㣺
        �Ӽ��˳���ת��(add, sub, mul, div, t)��
            z = x + y �� z = torch.add(x, y)
            y.add_(x)  # inplace�ͼӷ�

        ��numpyһ������Ƭ������
            x = y[:, 1]

        ����reshape
            x.view(new_shape) �� x = torch.randn(4, 4), x.view(16)��x.view(-1, 8)
"""
import torch

x = torch.empty(3, 5)
print(x)
y = torch.ones_like(x)
y.add_(x)
print(y)
z = torch.t(y)
print(z)
