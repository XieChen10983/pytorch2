# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/5/31 8:38
filename(文件名): training_process.py
function description(功能描述):
...
    1. define a neural network that has some learnable parameters(or weights).（定义模型）
        可用 net.parameters() 获取模型参数
    2. iterate over a dataset of inputs.（遍历整个输入）
    3. process input through the network.（通过模型计算每个输入）
    4. compute the loss.（根据模型的计算结果计算损失）
    5. propagate gradients back into the network's parameters.（根据损失backword()计算梯度）
    6. update the weights of the network: weight = weight - learning_rate * gradient（根据梯度更新模型参数）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1.定义网络并实例化
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, Input: torch.Tensor):
        x = F.max_pool2d(F.relu(self.conv1(Input)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        """
        平坦数据，即(batchsize, channel, height, width)型数据 => (batchsize, channel*height*width)型数据
        :param x:
        :return:
        """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)
# print(list(net.parameters()))

# 2.设置输入
input = torch.randn((1, 1, 32, 32))

# 3.通过网络计算输出
output = net(input)
print(output)

# 4.计算损失
target = torch.randn(10).view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)

# 5.计算梯度
with torch.no_grad():
    loss.backward()

# 6.更新参数
learning_rate = 0.01
for param in net.parameters():
    param.data.sub_(param.grad.data * learning_rate)
