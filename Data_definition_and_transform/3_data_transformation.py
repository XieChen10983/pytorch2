# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/5/30 15:45
filename(�ļ���): 3_data_transformation.py
function description(��������):
...
    ת��Ϊpython����
        ���tensor xֻ�е���Ԫ�أ���ô������x.item()�õ�python������
    ת��Ϊnumpy����
        y = x.numpy()
        ��ʱ����֮���������ģ�����x�����仯����y��֮�����仯��
    ��numpy����תΪtensor
        z = np.ones(3, 5)
        x = torch.from_numpy(z)
        ��ʱ����֮��Ҳ�������ģ�����numpy����z�����仯��tensor����xҲ��֮�仯
    ������ת��Ϊcuda���ݣ�
        �ж��Ƿ���cuda��
            torch.cuda.is_available()
        cpu����תΪcuda���ݣ�
            device = torch.device("cuda")
            x = torch.ones(3, 5, device=device)��
            x = torch.ones(3, 5), x = x.to(device)
"""
import torch
import numpy as np

x = torch.randn(1)
print(x)
print(x.item())
print(x.numpy())

y = np.ones((3, 5))
z = torch.from_numpy(y)
np.add(y, 1, out=y)
print(z)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(3, 5, device=device)
    print(x)
