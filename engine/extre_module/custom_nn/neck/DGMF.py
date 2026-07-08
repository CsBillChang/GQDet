"""Public DGMF interface used by GQDet."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DGMF(nn.Module):
    def __init__(self, channels, kernels=None):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("DGMF expects three feature levels.")
        out_ch = channels[1]
        kernels = kernels or [3, 5, 7]
        self.proj = nn.ModuleList(nn.Conv2d(c, out_ch, 1, bias=False) for c in channels)
        self.context = nn.ModuleList(
            nn.Conv2d(out_ch, out_ch, k, padding=k // 2, groups=out_ch, bias=False) for k in kernels[:3]
        )
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, 3, 1),
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, feats):
        target_size = feats[1].shape[-2:]
        aligned = []
        for feat, proj, context in zip(feats, self.proj, self.context):
            feat = proj(feat)
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="nearest")
            aligned.append(context(feat))
        weights = torch.softmax(self.gate(torch.cat(aligned, dim=1)), dim=1)
        fused = sum(weights[:, i : i + 1] * aligned[i] for i in range(3))
        return self.out(fused)
