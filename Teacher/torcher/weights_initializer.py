import torch.nn as nn


def dense_layers_init(net):
    """initializes dense layers - """
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
