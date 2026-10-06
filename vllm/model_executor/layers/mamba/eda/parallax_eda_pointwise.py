# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mesh Lambda preprocessing and output rounding points."""

import torch
import torch.nn.functional as F
from torch import nn


class LambdaRoPE(nn.Module):
    """Mesh rounds the four products before adding each rotary pair."""

    def __init__(self, rope, head_dim):
        super().__init__()
        self.register_buffer("cos_sin_cache", rope.cos_sin_cache, persistent=False)
        self.head_dim = head_dim

    def forward(self, positions, query, key=None):
        cos, sin = (
            self.cos_sin_cache.index_select(0, positions.long())
            .to(query.dtype)
            .chunk(2, -1)
        )
        cos, sin = cos[:, None], sin[:, None]

        def rotate(x):
            shape = x.shape
            x = x.reshape(x.shape[0], -1, self.head_dim)
            even, odd = x[..., 0::2], x[..., 1::2]
            a = even * cos - odd * sin
            b = even * sin + odd * cos
            return torch.stack((a, b), -1).flatten(-2).reshape(shape)

        return rotate(query), None if key is None else rotate(key)


class LambdaRMSNorm(nn.Module):
    """Keep FP32 gains while matching mesh's activation-dtype gain operands."""

    def __init__(self, weight, *, unit_gain=False, fp32_gain=False):
        super().__init__()
        self.weight = nn.Parameter(weight.float())
        self.eps = 1e-6
        self.unit_gain = unit_gain
        self.fp32_gain = fp32_gain

    def forward(self, x):
        if self.fp32_gain:
            xf = x.float()
            return (
                xf
                * xf.square().mean(-1, keepdim=True).add(self.eps).rsqrt()
                * self.weight.float()
            ).to(x.dtype)
        weight = self.weight.to(x.dtype)
        if self.unit_gain:
            weight = (
                weight.float() / weight.float().square().mean().add(self.eps).sqrt()
            ).to(x.dtype)
        return F.rms_norm(x, (x.shape[-1],), weight, self.eps).to(x.dtype)


def gates(mixed, erase, f, b, c, a_log, dt_bias, h, d):
    q, k, v = mixed.reshape(-1, 3, h, d).unbind(1)
    q, k, e = [
        F.normalize(x.float(), dim=-1).to(v.dtype)
        for x in (q, k, erase.reshape(-1, h, d))
    ]
    u = f.reshape(-1, h, d).float() + dt_bias.float().reshape(h, d)
    decay = (
        -5.0 + 5.0 * (-(a_log.float().exp().reshape(h, 1) / 5.0) * F.softplus(u)).exp()
    )
    return q, k, v.contiguous(), e, decay, b.float().sigmoid(), c.float().sigmoid()


def norm_gate(y, gate, weight):
    xf = y.float()
    return (
        xf
        / xf.square().mean(-1, keepdim=True).add(1e-6).sqrt()
        * weight.float()
        * F.silu(gate.float())
    ).to(y.dtype)
