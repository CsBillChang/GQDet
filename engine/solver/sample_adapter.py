"""Single-image sample helpers for the GQDet training loop."""

from __future__ import annotations

from typing import Any

import torch


def move_samples_to_device(samples: Any, device: torch.device, non_blocking: bool = True):
    if torch.is_tensor(samples):
        return samples.to(device, non_blocking=non_blocking)
    if isinstance(samples, dict):
        return {k: move_samples_to_device(v, device, non_blocking=non_blocking) for k, v in samples.items()}
    if isinstance(samples, list):
        return [move_samples_to_device(v, device, non_blocking=non_blocking) for v in samples]
    if isinstance(samples, tuple):
        return tuple(move_samples_to_device(v, device, non_blocking=non_blocking) for v in samples)
    return samples


def select_model_input(samples: Any, key: str = "image") -> torch.Tensor:
    if torch.is_tensor(samples):
        return samples
    if isinstance(samples, dict):
        for candidate in (key, "image", "images", "rgb"):
            value = samples.get(candidate)
            if torch.is_tensor(value):
                return value
        raise KeyError(f"No tensor input found in sample keys: {list(samples.keys())}")
    raise TypeError(f"Unsupported sample type: {type(samples)}")


def select_model_input_for_model(samples: Any, model: Any = None, key: str = "image"):
    return select_model_input(samples, key=key)


def select_plot_samples(samples: Any, key: str = "image"):
    return select_model_input(samples, key=key)


def select_plot_samples_for_logging(samples: Any, keys=("image", "rgb")):
    return [("image", select_plot_samples(samples))]
