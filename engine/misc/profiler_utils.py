"""Model profiling helpers."""

import copy
import os
import time
from typing import Tuple

import torch

from ..logger_module import get_logger

try:
    import thop
except Exception:
    thop = None

try:
    from calflops import calculate_flops
except Exception:
    calculate_flops = None

logger = get_logger(__name__)
RED, ORANGE, RESET = "\033[91m", "\033[38;5;208m", "\033[0m"


def stats(cfg, input_shape: Tuple = (1, 3, 640, 640), module=None) -> Tuple[int, dict]:
    input_shape = (1, 3, *cfg.yaml_cfg.get("eval_spatial_size", [640, 640]))
    model_for_info = copy.deepcopy(module if module is not None else cfg.model).deploy()
    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_for_info.to(device)

    flops, macs = "N/A", "N/A"
    if calculate_flops is not None:
        try:
            flops, macs, _ = calculate_flops(
                model=model_for_info,
                input_shape=input_shape,
                output_as_string=True,
                output_precision=4,
                print_detailed=False,
            )
        except Exception:
            logger.warning(RED + "calculate_flops failed; trying thop." + RESET)

    if (flops == "N/A" or macs == "N/A") and thop is not None:
        try:
            x = torch.randn(size=input_shape, device=device)
            macs_val = thop.profile(model_for_info, inputs=(x,), verbose=False)[0]
            macs, flops = thop.clever_format([macs_val, macs_val * 2], format="%.3f")
        except Exception:
            logger.warning(RED + "thop profile failed; FLOPs/MACs are unavailable." + RESET)

    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info
    return params, {f"Model FLOPs:{flops}   MACs:{macs}   Params:{params}"}


def get_weight_size(module):
    timestamp = f"{int(time.time() * 1000)}.pth"
    torch.save(module.state_dict(), timestamp)
    file_stats = os.stat(timestamp)
    logger.info(ORANGE + f"model size: {file_stats.st_size / (1024 ** 2):.1f} MB" + RESET)
    os.remove(timestamp)
