""" ResNext

ResNeXt-50: (3,4,6,3)
ResNeXt-101: (3,4,23,3)
ResNeXt-152: (3,8,36,3)

(Cardinality, bottleneck_width) = (1,64d), (2,40d), (4,24d), (8,14d), (32,4d)
"""

import torch
import torch.nn as nn
from Featurizer.classifier import Classifier

INPUT_SIZE = 224


class ResNext(Classifier):    
    class BottleneckBlock(nn.Module):
        """Depth=3 building block"""
        def __init__(self, in_planes, out_planes, stride, cardinality):
            super(ResNext.BottleneckBlock, self).__init__()
            self.shortcut_conv = None
            self.shortcut_bn = None
            if stride != 1 or in_planes != out_planes:
                self.shortcut_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
                self.shortcut_bn = nn.BatchNorm2d(out_planes)
            
            self.conv0 = nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(out_planes // 2)
            
            self.conv1 = nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=stride, groups=cardinality,
                                   bias=False, padding=1)
            self.bn1 = nn.BatchNorm2d(out_planes // 2)

            self.conv2 = nn.Conv2d(out_planes // 2, out_planes, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
            
            self.relu = nn.ReLU(inplace=True)

            nn.init.constant_(self.bn0.weight, 1)
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn2.weight, 0)
        
        def forward(self, x):
            out = self.conv0(x)
            out = self.bn0(out)
            out = self.relu(out)

            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            shortcut = x
            if self.shortcut_conv:
                shortcut = self.shortcut_conv(shortcut)
                shortcut = self.shortcut_bn(shortcut)
                
            out += shortcut
            out = self.relu(out)
            return out
        
    class NonBottleneckBlock(nn.Module):
        """Depth=2 building block"""
        def __init__(self):
            raise NotImplementedError

        def forward(self, x):
            pass

    def __init__(self, num_classes, num_stages=[3, 4, 6, 3], cardinality=32, bottleneck_width=4):
        in_planes = 64
        out_planes = bottleneck_width * cardinality * 2

        # Number of filters on the last conv layer. For example, it's 2048 for ResNeXt-50 32x4d.
        feature_planes = out_planes * (2 ** (len(num_stages) - 1))
        super(ResNext, self).__init__(num_classes, feature_planes)

        self.cardinality = cardinality
        
        # Build the stages
        stages = []
        first_block_stride = 1
        for n in num_stages:
            stages.append(self._make_stage(in_planes, out_planes, n, first_block_stride))
            in_planes = out_planes
            out_planes *= 2
            first_block_stride = 2
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            *stages,
            torch.nn.AdaptiveAvgPool2d(1)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def _make_stage(self, in_planes, out_planes, num_blocks, first_block_stride):
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResNext.BottleneckBlock(in_planes, out_planes, first_block_stride, self.cardinality))
            in_planes = out_planes
            first_block_stride = 1

        return nn.Sequential(*blocks)
