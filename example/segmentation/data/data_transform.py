# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/6/1 15:28
filename(文件名): data_transform.py
function description(功能描述):
...
    此代码对数据集进行变换，增强数据集
"""
import torchvision.transforms as T


def get_transform(train):
    """
    将图像转换为张量，当为训练模式时，对图像以0.5的概率进行水平翻转以增强；其他模式不用翻转。
    :param train:
    :return:
    """
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
