# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/5/31 9:56
filename(�ļ���): save_parameters.py
function description(��������):
    �˴�����ʾ��pytorchģ�͵ı��淽��֮һ���˷���������ģ�͵Ĳ�������������ģ�ͱ�������ĺ�׺��Ϊ.pkl��
    �����Ҫ���ñ����˵�ģ�Ͳ���ʱ����Ҫ�ȹ�����֮ƥ���ģ�͡�
        torch.save(net.state_dict(), PATH)
    ���뱣��Ĳ�����
        net = Model()
        net.load_state_dict(torch.load(PATH))
...
"""
import torch
import torch.nn as nn
from torch.optim import SGD


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 1, kernel_size=3)

    def forward(self, Input: torch.Tensor):
        x = self.conv1(Input)
        x = self.conv2(x)
        return x


net = Model()
net2 = Model()
optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)
input = torch.randn((1, 3, 32, 32))
output = net(input)

loss = torch.mean(output)

loss.backward()
optimizer.step()

params1 = list(net.parameters())
params2 = list(net2.parameters())
print(params1[0] == params2[0])  # ѵ��֮��net��net2�Ĳ�����һ��
# train the net.

torch.save(net.state_dict(), "./params.pkl")
net2.load_state_dict(torch.load("./params.pkl"))
print(list(net2.parameters())[0] == params1[0])  # ���뱣�������֮��net��net2�Ĳ���һ����
