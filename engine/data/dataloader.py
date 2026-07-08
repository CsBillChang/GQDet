"""
DEIM data-loading utilities used by the GQDet submission snapshot.

This file keeps the standard single-image detection training path and omits
unrelated experimental branches from the private development tree.
"""

import random

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import default_collate

from ..core import register
from ..logger_module import get_logger

logger = get_logger(__name__)

__all__ = [
    "DataLoader",
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "batch_image_collate_fn",
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ["dataset", "collate_fn"]

    def __repr__(self) -> str:
        rows = [self.__class__.__name__ + "("]
        for name in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            rows.append(f"    {name}: {getattr(self, name)}")
        rows.append(")")
        return "\n".join(rows)

    def set_epoch(self, epoch):
        self._epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        if hasattr(self.collate_fn, "set_epoch"):
            self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), "shuffle must be a boolean"
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    return torch.stack([x[0] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction:
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    def __call__(self, items):
        raise NotImplementedError


def generate_scales(base_size, base_size_repeat):
    if base_size_repeat is None:
        return None
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * int(base_size_repeat)
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        stop_epoch=None,
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
        mixup_prob=0.0,
        mixup_epochs=(0, 0),
        **kwargs,
    ):
        super().__init__()
        self.base_size = int(base_size)
        self.scales = generate_scales(self.base_size, base_size_repeat)
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        self.mixup_prob = float(mixup_prob)
        self.mixup_epochs = list(mixup_epochs)
        self.extra_cfg = kwargs
        if self.scales is not None:
            logger.info(f"Multi-scale training enabled until epoch {self.stop_epoch}.")

    def _collate_targets(self, items):
        return [x[1] for x in items]

    def _resize_batch(self, images):
        if self.scales is None or self.epoch >= self.stop_epoch:
            return images
        size = random.choice(self.scales)
        return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)

    def __call__(self, items):
        images = torch.stack([x[0] for x in items], dim=0)
        targets = self._collate_targets(items)
        images = self._resize_batch(images)
        return images, targets


def simple_collate(items):
    return default_collate(items)
