# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/6/1 9:41
filename(文件名): reshape_module.py
function description(功能描述):
...
    1. 此代码演示如何构建自己的功能模块，例如针对mnist数据集28*28，由于pytorch中没有reshape模块，需要自己构建。
    2. 构建自己的模块需要继承nn.Module模块，并实现其中的forward方式，反向传播会自动完成，不用自己实现。
    3. 在构建好自己的功能模块之后，可以像nn中的各个模块一样使用。
"""
import torch
import torch.nn as nn


class Reshape(nn.Module):
    """
    此类的作用为给定一个reshape方程，将输入按照该方程reshape之后输出。
    """
    def __init__(self, reshape_func):
        super(Reshape, self).__init__()
        self.reshape_func = reshape_func

    def forward(self, Input):
        return self.reshape_func(Input)


if __name__ == "__main__":
    net = nn.Sequential(
        Reshape(lambda x: x.view(-1, 1, 28, 28)),  # 给定一个reshape方程，构建一个Reshape实例。
        nn.Conv2d(1, 16, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(16, 16, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(16, 10, 3, 2, 1),
        nn.ReLU(),
        nn.AvgPool2d(4),
        Reshape(lambda x: x.view(x.size(0), -1)),  # 给定一个reshape方程，构建一个Reshape实例
    )

    input = torch.randn(10, 784)
    output = net(input)
    print(output.size())
