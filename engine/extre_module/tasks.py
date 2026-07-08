import contextlib
import re
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from ..backbone.common import FrozenBatchNorm2d
from ..backbone.hgnetv2 import HG_Stage, StemBlock, filter_loadable_state_dict
from ..core import register
from ..deim.gqrc_dfine_decoder import GQRCDFINETransformer
from ..deim.hybrid_encoder import ConvNormLayer_fuse, RepNCSPELAN4
from ..extre_module.custom_nn.neck.DGMF import DGMF
from ..extre_module.custom_nn.transformer.DeformableSelfAttention import TransformerEncoderLayer_DeformableSelfAttention
from ..extre_module.ultralytics_nn.conv import Concat
from ..logger_module import get_logger
from ..misc.dist_utils import is_dist_available_and_initialized

logger = get_logger(__name__)

__all__ = ["DEIM_MG"]


def yaml_load(file="data.yaml"):
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file}"
    with open(file, errors="ignore", encoding="utf-8") as f:
        text = f.read()
    if not text.isprintable():
        text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", text)
    return yaml.safe_load(text) or {}


def _get_channels(channels, f, current):
    if isinstance(f, list):
        return [_get_channels(channels, item, current) for item in f]
    if f == -1:
        return current
    return channels[f]


def _resolve_module(name):
    if isinstance(name, str) and name.startswith("nn."):
        return getattr(nn, name[3:])
    table = {
        "StemBlock": StemBlock,
        "HG_Stage": HG_Stage,
        "ConvNormLayer_fuse": ConvNormLayer_fuse,
        "RepNCSPELAN4": RepNCSPELAN4,
        "TransformerEncoderLayer_DeformableSelfAttention": TransformerEncoderLayer_DeformableSelfAttention,
        "DGMF": DGMF,
        "Concat": Concat,
        "GQRCDFINETransformer": GQRCDFINETransformer,
    }
    if name not in table:
        raise KeyError(f"Module {name} is not exposed in the public GQDet parser.")
    return table[name]


def parse_module(i, f, m, args, channels, current, num_classes=None, eval_spatial_size=None):
    module = _resolve_module(m)
    args = [] if args is None else list(args) if isinstance(args, list) else dict(args)
    in_ch = _get_channels(channels, f, current)

    if module is StemBlock:
        build_args = [in_ch, args[0], args[1], *args[2:]]
        out_ch = args[1]
    elif module is HG_Stage:
        build_args = [in_ch, args[0], args[1], *args[2:]]
        out_ch = args[1]
    elif module in {ConvNormLayer_fuse, RepNCSPELAN4}:
        build_args = [in_ch, *args]
        out_ch = args[0]
    elif module is TransformerEncoderLayer_DeformableSelfAttention:
        build_args = [in_ch, *args]
        out_ch = in_ch
    elif module is DGMF:
        build_args = [in_ch, *args]
        out_ch = in_ch[1]
    elif module is Concat:
        build_args = args
        out_ch = sum(in_ch)
    elif module is nn.Upsample:
        build_args = args
        out_ch = in_ch
    elif module is GQRCDFINETransformer:
        build_args = dict(args)
        build_args["feat_channels"] = in_ch
        build_args["num_classes"] = num_classes
        build_args["eval_spatial_size"] = eval_spatial_size
        out_ch = num_classes
    else:
        build_args = args
        out_ch = in_ch

    layer = module(**build_args) if isinstance(build_args, dict) else module(*build_args)
    layer.np = sum(x.numel() for x in layer.parameters())
    layer.i, layer.f, layer.type = i, f, module.__name__
    return layer, out_ch


def parse_model(d, ch, nc, eval_spatial_size, verbose=True):
    channels, current = [], ch
    layer_index = 0
    backbone_layers, encoder_layers, save = [], [], []
    decoder_model = None

    for section, target in (("backbone", backbone_layers), ("encoder", encoder_layers)):
        for f, m, args in d.get(section, []):
            layer, current = parse_module(layer_index, f, m, args, channels, current, eval_spatial_size=eval_spatial_size)
            save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)
            target.append(layer)
            channels.append(current)
            layer_index += 1

    for f, m, args in d.get("decoder", []):
        decoder_model, current = parse_module(
            layer_index, f, m, args, channels, current, num_classes=nc, eval_spatial_size=eval_spatial_size
        )
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)
        channels.append(current)

    return nn.Sequential(*backbone_layers), nn.Sequential(*encoder_layers), decoder_model, sorted(set(save))


@register(force=True)
class DEIM_MG(nn.Module):
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        yaml_path,
        pretrained=None,
        freeze_stem_only=False,
        freeze_at=-1,
        freeze_norm=False,
        num_classes=80,
        eval_spatial_size=(640, 640),
    ):
        super().__init__()
        model_cfg = yaml_load(yaml_path)
        self.backbone, self.encoder, self.decoder, self.save = parse_model(
            model_cfg, ch=3, nc=num_classes, eval_spatial_size=eval_spatial_size
        )

        if freeze_at >= 0:
            self._freeze_parameters(self.backbone[0])
        if freeze_norm:
            self._freeze_norm(self.backbone)

        if pretrained:
            try:
                state = torch.load(pretrained, map_location="cpu")
                filtered_state = filter_loadable_state_dict(self.backbone.state_dict(), state)
                self.backbone.load_state_dict(filtered_state, strict=False)
            except Exception as exc:
                if (is_dist_available_and_initialized() and torch.distributed.get_rank() == 0) or (
                    not is_dist_available_and_initialized()
                ):
                    logger.error(f"Loading backbone pretrained weight failed: {exc}")
                raise

    def forward(self, x, targets=None):
        y = []
        for m in list(self.backbone.children()) + list(self.encoder.children()):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return self.decoder([y[j] for j in self.decoder.f], targets)

    def deploy(self):
        self.eval()
        for m in self.modules():
            with contextlib.suppress(AttributeError):
                m.convert_to_deploy()
        return self

    def _freeze_norm(self, module):
        if isinstance(module, nn.BatchNorm2d):
            return FrozenBatchNorm2d(module.num_features)
        for name, child in module.named_children():
            new_child = self._freeze_norm(child)
            if new_child is not child:
                setattr(module, name, new_child)
        return module

    def _freeze_parameters(self, module):
        for p in module.parameters():
            p.requires_grad = False
