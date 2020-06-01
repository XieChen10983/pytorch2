# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/6/1 15:12
filename(文件名): finetuning_model.py
function description(功能描述):
...
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
