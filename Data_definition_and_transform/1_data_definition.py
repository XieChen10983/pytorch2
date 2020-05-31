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

    ��ȡ���ݶȵ�������requires_grad=True �� x.requires_grad_(True), ֻ���ݶ������������ܽ���back-prop
        x = torch.Tensor([3, 5], requires_grad=True)
    �����ݶȵı������ɵı���Ҳ�����ݶȵġ�
        y = x*x  =>  y.requires_grad = True

    �����ݶȵı������м���ʱ����ñ�����ص����б����ݶȶ��ᷢ���仯��Ϊ�˱������������
        ���ڵ�������x��x.detach() �� x.requires_grad_(False)
        ����һ���δ����еı���: with torch.no_grad():

    �����ݶȵļ��㣺
        scalar.backward()
    �����ݶȵĻ�ȡ��
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
