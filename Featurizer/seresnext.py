import torch.nn as nn
from Featurizer.resnext import ResNext


class SEResNext(ResNext):
    class SEBlock(nn.Module):
        def __init__(self, in_planes, reduction_ratio):
            super(SEResNext.SEBlock, self).__init__()
            self.in_planes = in_planes
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.fc0 = nn.Linear(in_planes, in_planes // reduction_ratio)
            self.relu = nn.ReLU(inplace=True)
            self.fc1 = nn.Linear(in_planes // reduction_ratio, in_planes)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            scale = self.pooling(x)
            scale = scale.view(-1, self.in_planes)
            scale = self.fc0(scale)
            scale = self.relu(scale)
            scale = self.fc1(scale)
            scale = self.sigmoid(scale)
            scale = scale.view(-1, self.in_planes, 1, 1)
            return x * scale

    class BottleneckBlock(ResNext.BottleneckBlock):
        def __init__(self, in_planes, out_planes, stride, cardinality, reduction_ratio):
            super(SEResNext.BottleneckBlock, self).__init__(in_planes, out_planes, stride, cardinality)
            self.seblock = SEResNext.SEBlock(out_planes, reduction_ratio)

        def forward(self, x):
            out = self.conv0(x)
            out = self.bn0(out)
            out = self.relu(out)

            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out = self.seblock(out)
            
            shortcut = x
            if self.shortcut_conv:
                shortcut = self.shortcut_conv(shortcut)
                shortcut = self.shortcut_bn(shortcut)
                
            out += shortcut
            out = self.relu(out)
            return out

    def __init__(self, num_classes, num_stages=[3, 4, 6, 3], cardinality=32, bottleneck_width=4, reduction_ratio=16):
        self.reduction_ratio = reduction_ratio
        super(SEResNext, self).__init__(num_classes, num_stages, cardinality, bottleneck_width)
        
    def _make_stage(self, in_planes, out_planes, num_blocks, first_block_stride):
        blocks = []
        for i in range(num_blocks):
            blocks.append(SEResNext.BottleneckBlock(in_planes, out_planes, first_block_stride, self.cardinality,
                                                    self.reduction_ratio))
            in_planes = out_planes
            first_block_stride = 1

        return nn.Sequential(*blocks)
