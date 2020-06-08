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

        ��ά����˷�(mm) �� @��
            x.mm(y) �� x @ y
        ��ά����˷�(��һά��batchsize)��torch.bmm(x, y)
        ��Ͼ���˷�(��Ϊ���ӣ�����鿴pytorch�̳�)��torch.matmul(x, y)
        ���˷���torch.mul(x, y)

        ��numpyһ������Ƭ������
            x = y[:, 1]

        ����reshape
            x.view(new_shape) �� x = torch.randn(4, 4), x.view(16)��x.view(-1, 8)
"""
import torch

x = torch.empty(3, 5)
print(x)
y = torch.ones_like(x)
# �Լӷ�
y.add_(x)
print(y)

# ����˷�
x1 = torch.randint(1, 4, (3, 5))
x2 = torch.randint(1, 4, (5, 3))
output = x1.mm(x2)
print(output.size())
output2 = x1 @ x2
print(output2.size())

# ת��
z = torch.t(y)
print(z)
