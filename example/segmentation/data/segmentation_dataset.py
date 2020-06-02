# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/6/1 11:46
filename(文件名): segmentation_dataset.py
function description(功能描述):
...
    此代码设置图像分割的数据集
"""
import os
import numpy as np
import torch
from PIL import Image
# from torch.utils.data import DataLoader


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, item):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[item])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[item])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)  # unique函数去除重复数字并且排序输出。
        ojb_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        num_objs = len(ojb_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs, ), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([item])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2], boxes[:, 0])

        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id,
                  "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
