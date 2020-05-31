# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/5/31 8:38
filename(�ļ���): training_process.py
function description(��������):
...
    1. define a neural network that has some learnable parameters(or weights).������ģ�ͣ�
        ���� net.parameters() ��ȡģ�Ͳ���
    2. iterate over a dataset of inputs.�������������룩
    3. process input through the network.��ͨ��ģ�ͼ���ÿ�����룩
    4. compute the loss.������ģ�͵ļ�����������ʧ��
    5. propagate gradients back into the network's parameters.��������ʧbackword()�����ݶȣ�
    6. update the weights of the network: weight = weight - learning_rate * gradient�������ݶȸ���ģ�Ͳ�����
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1.�������粢ʵ����
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
        ƽ̹���ݣ���(batchsize, channel, height, width)������ => (batchsize, channel*height*width)������
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

# 2.��������
input = torch.randn((1, 1, 32, 32))

# 3.ͨ������������
output = net(input)
print(output)

# 4.������ʧ
target = torch.randn(10).view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)

# 5.�����ݶ�
with torch.no_grad():
    loss.backward()

# 6.���²���
learning_rate = 0.01
for param in net.parameters():
    param.data.sub_(param.grad.data * learning_rate)
