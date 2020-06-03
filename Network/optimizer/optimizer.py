# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/6/1 7:45
filename(文件名): optimizer.py
function description(功能描述):
...
    一. 自定义的优化过程流程是这样：
        1. 计算出损失loss
        2. model.zero_grad(), loss.backward()反向传播自动计算每个节点梯度。
        3. 梯度下降法更新每个参数：
            with torch.no_grad():
                for param in model.parameters():
                    param.sub_(learning_rate * param.grad)

    二. 以上过程可以通过torch.optim中的各个优化器（例如SGD）进行:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        ...
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    三. 优化器中的参数：
        优化器的第一个参数为需要更新的模型的参数，例如只需要更新最后的全连接层时：
            optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)

    四. 变学习率的优化器：
        使用optim模块中的lr_scheduler函数：
            scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma)
        其中optimizer为模型的优化器，epochs为学习率变化的间隔，如step_sizes=30表示每隔30个epoch学习率更新一次；
        gamma为学习率变化的大小，如gamma=0.1表示学习率变为原来的0.1。

        (1) 用法
            ...
            optimizer.step()
            scheduler.step()
        (2) 变化类型
            a. lr_scheduler.StepLR(...)每隔一定的epoch就有一定的变化
            b. lr_scheduler.MultiStepLR(optimizer, [low, high], gamma)在epoch=low和high的时候变化
            c. lr_scheduler.ExponentialLR(optimizer, gamma)以gamma为系数的指数型变化。
"""
import torch
from time import time

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)

start = time()
for _ in range(1):
    x_ = []
    y_ = []
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        scheduler.step(epoch=None)
        x_.append(t)
        y_.append(scheduler.get_lr()[0])

    import matplotlib.pyplot as plt

    plt.plot(x_, y_)
    plt.show()

end = time()
print("running time: ", end - start)
