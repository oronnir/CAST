import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidCrossEntropyLossWithBalancing(nn.Module):
    """good for classification """
    def __init__(self, class_wise_negative_sample_weights):
        super(SigmoidCrossEntropyLossWithBalancing, self).__init__()
        self.class_wise_negative_sample_weights = torch.from_numpy(class_wise_negative_sample_weights).float().cuda()
        self.norm = self.class_wise_negative_sample_weights.mean().item()

    def forward(self, input_t, target):
        weights = torch.ones_like(target) * self.class_wise_negative_sample_weights
        weights = weights + target
        weights = torch.clamp(weights, 0.0, 1.0)
        return F.binary_cross_entropy_with_logits(input_t, target, weights)/self.norm
