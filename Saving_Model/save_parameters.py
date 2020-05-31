# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/5/31 9:56
filename(文件名): save_parameters.py
function description(功能描述):
    此代码演示了pytorch模型的保存方法之一，此方法仅保存模型的参数，而不保存模型本身，保存的后缀名为.pkl。
    因此需要利用保存了的模型参数时，需要先构建与之匹配的模型。
        torch.save(net.state_dict(), PATH)
    载入保存的参数：
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
print(params1[0] == params2[0])  # 训练之后，net和net2的参数不一样
# train the net.

torch.save(net.state_dict(), "./params.pkl")
net2.load_state_dict(torch.load("./params.pkl"))
print(list(net2.parameters())[0] == params1[0])  # 载入保存的数据之后，net和net2的参数一样了
