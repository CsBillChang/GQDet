"""Public DSWA interface used by GQDet.

The released implementation preserves the module signature and insertion point
while keeping the detailed research variant private during review.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DeformableSelfAttention", "TransformerEncoderLayer_DeformableSelfAttention"]


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DeformableSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=7, dilation=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.local = nn.Conv2d(dim, dim, kernel_size, padding=padding, dilation=dilation, groups=dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.drop = nn.Dropout(proj_drop)
        self.scale = (dim // max(num_heads, 1)) ** -0.5
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        global_ctx = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return self.drop(self.proj(global_ctx + self.local(x)))


class TransformerEncoderLayer_DeformableSelfAttention(nn.Module):
    def __init__(
        self,
        c1,
        cm=2048,
        num_heads=8,
        kernel_size=7,
        dilation=1,
        dropout=0.0,
        act=nn.GELU(),
        normalize_before=False,
    ):
        super().__init__()
        self.dswa = DeformableSelfAttention(c1, num_heads=num_heads, kernel_size=kernel_size, dilation=dilation)
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)
        self.norm1 = LayerNorm2d(c1)
        self.norm2 = LayerNorm2d(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src = self.norm1(src + self.dropout1(self.dswa(src)))
        src = self.norm2(src + self.dropout2(self.fc2(self.act(self.fc1(src)))))
        return src
