# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/6/1 7:45
filename(�ļ���): optimizer.py
function description(��������):
...
    һ. �Զ�����Ż�����������������
        1. �������ʧloss
        2. model.zero_grad(), loss.backward()���򴫲��Զ�����ÿ���ڵ��ݶȡ�
        3. �ݶ��½�������ÿ��������
            with torch.no_grad():
                for param in model.parameters():
                    param.sub_(learning_rate * param.grad)

    ��. ���Ϲ��̿���ͨ��torch.optim�еĸ����Ż���������SGD������:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        ...
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ��. �Ż����еĲ�����
        �Ż����ĵ�һ������Ϊ��Ҫ���µ�ģ�͵Ĳ���������ֻ��Ҫ��������ȫ���Ӳ�ʱ��
            optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)

    ��. ��ѧϰ�ʵ��Ż�����
        ʹ��optimģ���е�lr_scheduler������
            scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma)
        ����optimizerΪģ�͵��Ż�����epochsΪѧϰ�ʱ仯�ļ������step_sizes=30��ʾÿ��30��epochѧϰ�ʸ���һ�Σ�
        gammaΪѧϰ�ʱ仯�Ĵ�С����gamma=0.1��ʾѧϰ�ʱ�Ϊԭ����0.1��

        (1) �÷�
            ...
            optimizer.step()
            scheduler.step()
        (2) �仯����
            a. lr_scheduler.StepLR(...)ÿ��һ����epoch����һ���ı仯
            b. lr_scheduler.MultiStepLR(optimizer, [low, high], gamma)��epoch=low��high��ʱ��仯
            c. lr_scheduler.ExponentialLR(optimizer, gamma)��gammaΪϵ����ָ���ͱ仯��
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
