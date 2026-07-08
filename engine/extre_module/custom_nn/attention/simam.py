import torch.nn as nn


class SimAM(nn.Module):
    def __init__(self, channels=None, *args, **kwargs):
        super().__init__()
        self.op = nn.Identity()

    def forward(self, x):
        return self.op(x)
