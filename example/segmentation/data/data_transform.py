# coding=gbk
"""
author(����): Channing Xie(л�)
time(ʱ��): 2020/6/1 15:28
filename(�ļ���): data_transform.py
function description(��������):
...
    �˴�������ݼ����б任����ǿ���ݼ�
"""
import torchvision.transforms as T


def get_transform(train):
    """
    ��ͼ��ת��Ϊ��������Ϊѵ��ģʽʱ����ͼ����0.5�ĸ��ʽ���ˮƽ��ת����ǿ������ģʽ���÷�ת��
    :param train:
    :return:
    """
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
