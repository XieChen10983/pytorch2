# coding=gbk
"""
author(作者): Channing Xie(谢琛)
time(时间): 2020/6/1 15:16
filename(文件名): different_backbone.py
function description(功能描述):
...
"""
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512), ),
                                   aspect_ratios=((0.5, 1.0, 2.0), ))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

