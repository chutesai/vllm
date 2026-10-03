# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental BF16 shared-expert GEMM, FP32 accumulation and BF16 round points."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _linear(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    XS: tl.constexpr,
    WS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    RELU2: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        kk = base * BK + k
        x = tl.load(
            X + m[:, None] * XS + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0
        )
        w = tl.load(
            W + n[None, :] * WS + kk[:, None], (n[None, :] < N) & (kk[:, None] < K), 0
        )
        acc = tl.dot(x, w, acc)
    value = acc.to(tl.bfloat16)
    if RELU2:
        value = tl.maximum(value.to(tl.float32), 0)
        value = (value * value).to(tl.bfloat16)
    tl.store(
        Y + m[:, None] * N + n[None, :], value, (m[:, None] < M) & (n[None, :] < N)
    )


def linear(x, weight, *, relu2=False, tile=(64, 64, 64, 4, 3)):
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("BF16 inputs and weights required")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Contiguous matrices required")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("Expected matching GEMM reduction widths")
    m, k = x.shape
    n = weight.shape[0]
    bm, bn, bk, warps, stages = tile
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _linear[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x,
        weight,
        out,
        m,
        n,
        k,
        x.stride(0),
        weight.stride(0),
        bm,
        bn,
        bk,
        relu2,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
    return out
