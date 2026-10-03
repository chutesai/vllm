# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Standalone BAR norm; same forward operands as the training RMSNorm."""

import torch
import torch.nn.functional as F
from torch import nn


def gain_for(weight, dtype):
    return weight if weight.dtype == dtype else weight.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        weight = gain_for(self.weight, x.dtype)
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), weight, self.eps).to(x.dtype)
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * weight
