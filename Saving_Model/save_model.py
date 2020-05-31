# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/5/31 10:34
filename(�ļ���): save_model.py
function description(��������):
    �˴�����ʾ��pytorchģ�͵ı��淽��֮һ���˷���������ģ�ͱ�������ĺ�׺��Ϊ.pkl��
    ��Ҫ���ñ����˵�ģ��ʱ��ֱ����������ģ�͡�
        torch.save(net, PATH)
    ���뱣��Ĳ�����
        net = torch.load(PATH)
...
"""
import torch
import torch.nn as nn


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
input = torch.randn((1, 3, 32, 32))
output = net(input)

loss = torch.mean(output)
print(loss)

loss.backward()
net.step()

params1 = list(net.parameters())
params2 = list(net2.parameters())
# train the net.

torch.save(net, "./model.pkl")
net2 = torch.load("./model.pkl")
