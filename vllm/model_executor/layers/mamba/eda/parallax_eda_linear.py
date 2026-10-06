# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Joined EDA GEMM with explicit FP32 reduction and bias before BF16 storage."""

import torch
from torch import nn

from vllm.triton_utils import tl, triton


@triton.jit
def _linear(
    X,
    W,
    B,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        kk = base * BK + k
        x = tl.load(
            X + m[:, None] * XS + kk[None, :],
            (m[:, None] < M) & (kk[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + n[None, :] * WS + kk[:, None],
            (n[None, :] < N) & (kk[:, None] < K),
            other=0,
        )
        acc = tl.dot(x, w, acc)
    bias = tl.load(B + n, n < N, other=0).to(tl.float32)
    tl.store(
        Y + m[:, None] * N + n[None, :],
        acc + bias[None, :],
        (m[:, None] < M) & (n[None, :] < N),
    )


def linear(x, weight, bias, *, tile=(32, 64, 64, 4, 3)):
    """Retain row/input strides and round once after the FP32 bias epilogue."""
    m, k = x.shape
    n = weight.shape[0]
    bm, bn, bk, warps, stages = tile
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _linear[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x,
        weight,
        bias,
        out,
        m,
        n,
        k,
        x.stride(0),
        weight.stride(0),
        bm,
        bn,
        bk,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
    return out


class StableJoinedEDAProjection(nn.Linear):
    def forward(self, x):
        if x.shape[0] >= 256:
            return super().forward(x)
        return linear(x, self.weight, self.bias, tile=self.tile)

    @classmethod
    def from_joined(cls, joined, tile=(32, 64, 64, 4, 3)):
        result = cls(joined.in_features, joined.out_features, bias=True, device="meta")
        result.weight, result.bias = joined.weight, joined.bias
        result.tile = tile
        return result
