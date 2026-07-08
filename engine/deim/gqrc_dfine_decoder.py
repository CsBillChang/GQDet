"""Public GQRC decoder interface for GQDet."""

import torch
import torch.nn as nn

from .dfine_decoder import DFINETransformer


class GQRCDFINETransformer(DFINETransformer):
    def __init__(
        self,
        *args,
        gqrc_scale=0.20,
        gqrc_hidden_dim=64,
        gqrc_detach_boxes=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gqrc_scale = float(gqrc_scale)
        self.gqrc_detach_boxes = bool(gqrc_detach_boxes)
        hidden_dim = kwargs.get("hidden_dim", getattr(self, "hidden_dim", 256))
        self.gqrc_head = nn.Sequential(
            nn.Linear(4, gqrc_hidden_dim),
            nn.SiLU(),
            nn.Linear(gqrc_hidden_dim, 1),
        )
        self.gqrc_proj = nn.Linear(1, self.num_classes)

    def forward(self, feats, targets=None):
        out = super().forward(feats, targets=targets)
        boxes = out["pred_boxes"].detach() if self.gqrc_detach_boxes else out["pred_boxes"]
        quality = self.gqrc_proj(self.gqrc_head(boxes))
        out["pred_logits"] = out["pred_logits"] + self.gqrc_scale * quality
        if self.training:
            out["gqrc_quality"] = torch.sigmoid(quality)
        return out
