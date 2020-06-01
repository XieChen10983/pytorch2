# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/6/1 7:09
filename(文件名): linear_operation.py
function description(功能描述):
...
    此代码演示如何用pytorch定义一个线形计算层以及直接使用自带的api

"""
import torch
import torch.nn as nn
from self_defined_relu import MyReLU


# 用nn自带的线性层进行定义
class LinearNet(nn.Module):
    """
    这一个类取代了下面代码中线性层的定义，即以下语句
        w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
        w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
        relu = MyReLU.apply
    并表明了计算顺序等关系。
    """
    def __init__(self, D_IN, HIDDEN, D_OUT):
        super(LinearNet, self).__init__()
        self.D_IN = D_IN
        self.D_OUT = D_OUT
        self.HIDDEN = HIDDEN
        self.fc1 = nn.Linear(self.D_IN, self.HIDDEN)
        self.fc2 = nn.Linear(self.HIDDEN, self.D_OUT)
        self.relu = nn.ReLU()

    def forward(self, INPUT):
        return self.fc2(self.relu(self.fc1(INPUT)))


###################################################################################
dtype = torch.float
device = torch.device("cpu")
N, D_in, H, D_out = 64, 1000, 100, 10  # 分别为batchsize、输入维度、隐藏层、输出维度
learning_rate = 1e-6

x = torch.randn(N, D_in, dtype=dtype, device=device)
y = torch.randn(N, D_out, dtype=dtype, device=device)


# 自定义线性层：
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
relu = MyReLU.apply

print("测试自定义的线性层计算")
for t in range(500):
    y_pred = relu(x.mm(w1)).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()

#####################################################################################
print("测试自带的线性层计算")
net = LinearNet(D_in, H, D_out)
for t in range(20000):
    y_pred = net(x)
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
    net.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in net.parameters():
            param.sub_(learning_rate * param.grad)
            # param -= learning_rate * param.grad
