"""Lightweight placeholders for optional distillation paths."""

import torch.nn as nn


class FeatureLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return 0.0


class DETRLogicLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return 0.0


class DETRMutilDecoderLogicLoss(DETRLogicLoss):
    pass
