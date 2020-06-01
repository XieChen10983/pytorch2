# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/6/1 9:41
filename(�ļ���): reshape_module.py
function description(��������):
...
    1. �˴�����ʾ��ι����Լ��Ĺ���ģ�飬�������mnist���ݼ�28*28������pytorch��û��reshapeģ�飬��Ҫ�Լ�������
    2. �����Լ���ģ����Ҫ�̳�nn.Moduleģ�飬��ʵ�����е�forward��ʽ�����򴫲����Զ���ɣ������Լ�ʵ�֡�
    3. �ڹ������Լ��Ĺ���ģ��֮�󣬿�����nn�еĸ���ģ��һ��ʹ�á�
"""
import torch
import torch.nn as nn


class Reshape(nn.Module):
    """
    ���������Ϊ����һ��reshape���̣������밴�ո÷���reshape֮�������
    """
    def __init__(self, reshape_func):
        super(Reshape, self).__init__()
        self.reshape_func = reshape_func

    def forward(self, Input):
        return self.reshape_func(Input)


if __name__ == "__main__":
    net = nn.Sequential(
        Reshape(lambda x: x.view(-1, 1, 28, 28)),  # ����һ��reshape���̣�����һ��Reshapeʵ����
        nn.Conv2d(1, 16, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(16, 10, 3, 2, 1),
        nn.ReLU(),
        nn.AvgPool2d(4),
        Reshape(lambda x: x.view(x.size(0), -1)),  # ����һ��reshape���̣�����һ��Reshapeʵ��
    )

    input = torch.randn(10, 784)
    output = net(input)
    print(output.size())
