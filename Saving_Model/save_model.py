# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/5/31 10:34
filename(文件名): save_model.py
function description(功能描述):
    此代码演示了pytorch模型的保存方法之一，此方法保存了模型本身，保存的后缀名为.pkl。
    需要利用保存了的模型时，直接载入整个模型。
        torch.save(net, PATH)
    载入保存的参数：
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
