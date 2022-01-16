import torch.nn as nn


class ResNetWithFeatures(nn.Module):

    def __init__(self, model):
        super(ResNetWithFeatures, self).__init__()
        layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3",
                  "layer4", "avgpool", "fc"]

        for l_name in layers:
            setattr(self, l_name, getattr(model, l_name))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        self.fea = x

        x = self.fc(x)

        return x, self.fea
    
    @property
    def features(self):
        return self.fea

