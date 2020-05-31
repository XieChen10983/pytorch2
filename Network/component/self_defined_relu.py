# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/5/31 17:24
filename(�ļ���): self_defined_relu.py
function description(��������):
    �˴�����ʾ������Զ���һ��autograd�����࣬���ﶨ�����һ��relu������
        1. ��Ҫ�̳�torch.autograd.Function
        2. ������Ҫ���forward��backward��������
        3. �������֮�����MyReLU.apply���õ�һ������
        4. �ú���������pytorch�Դ���autograd����һ������һ������������Ĺ����У�Ҳ���Ե���һ��������
            ������Զ������ݶ���Ϣ��
...
"""
import torch
import torch.nn as nn


# �˶δ�����ʾ���Զ����autograd operation��������������Ĺ�����
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.relu1 = MyReLU.apply
        self.conv2 = nn.Conv2d(6, 10, 3)

    def forward(self, Input):
        output = self.conv1(Input)
        output = self.relu1(output)
        output = self.conv2(output)
        return output


# �˶δ�����ʾ������Զ���һ��autograd operation
class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    # apply = None

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


if __name__ == "__main__":
    dtype = torch.float
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold input and outputs.
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Create random Tensors for weights.
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        # To apply our Function, we use Function.apply method. We alias this as 'relu'.
        relu = MyReLU.apply

        # Forward pass: compute predicted y using operations; we compute
        # ReLU using our custom autograd operation.
        y_pred = relu(x.mm(w1)).mm(w2)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())

        # Use autograd to compute the backward pass.
        loss.backward()

        # Update weights using gradient descent
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()

    net = MyModel()
    in_ = torch.randn(10, 3, 32, 32)
    out = net(in_)
    print(out.size())
