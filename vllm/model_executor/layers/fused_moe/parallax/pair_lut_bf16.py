# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental FP32 pair tables with BF16 activation round points.

This computes the same real-valued ternary products, but factors the row
scale outside the sum. It requires numerical qualification before serving.
"""

import torch

from vllm.triton_utils import tl, triton

from .ternary_decode import _combine


@triton.jit
def _tables(X, Y, ROWS: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    pos = tl.program_id(0) * B + tl.arange(0, B)
    pair = pos // 9
    pattern = pos % 9
    a, b = pattern % 3, pattern // 3
    x0 = tl.load(X + pair * 2, pair < ROWS * K // 2, other=0).to(tl.float32)
    x1 = tl.load(X + pair * 2 + 1, pair < ROWS * K // 2, other=0).to(tl.float32)
    c0 = (a == 1).to(tl.float32) - (a == 2).to(tl.float32)
    c1 = (b == 1).to(tl.float32) - (b == 2).to(tl.float32)
    tl.store(Y + pos, c0 * x0 + c1 * x1, pos < ROWS * K // 2 * 9)


@triton.jit
def _apply(
    TABLE,
    W,
    META,
    ALPHA,
    IDS,
    GATES,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    TOP: tl.constexpr,
    FIRST: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DIRECT: tl.constexpr,
):
    route = tl.program_id(0)
    source = route // TOP if FIRST else route
    expert = tl.load(IDS + route)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    blocks = tl.arange(0, BK)
    s = tl.load(
        W + (expert * N + ns[:, None]) * (K // 8) + blocks[None, :],
        (ns[:, None] < N) & (blocks[None, :] < K // 8),
        other=0,
    ).to(tl.int32)
    meta = tl.load(
        META + (expert * N + ns[:, None]) * (K // 16) + blocks[None, :] // 2,
        (ns[:, None] < N) & (blocks[None, :] < K // 8),
        other=0,
    ).to(tl.int32)
    nibble = (meta >> ((blocks[None, :] % 2) * 8)) & 15
    p0, p1 = nibble & 3, (nibble >> 2) & 3
    q0 = (s & 3) + 3 * ((s >> 4) & 3)
    q1 = ((s >> 2) & 3) + 3 * ((s >> 6) & 3)
    base = (source * (K // 2) + blocks[None, :] * 4) * 9
    mask = (ns[:, None] < N) & (blocks[None, :] < K // 8)
    if DIRECT:
        offset0 = source * K + blocks[None, :] * 8 + p0 * 2
        offset1 = source * K + blocks[None, :] * 8 + p1 * 2
        x0 = tl.load(TABLE + offset0, mask, other=0).to(tl.float32)
        x1 = tl.load(TABLE + offset0 + 1, mask, other=0).to(tl.float32)
        x2 = tl.load(TABLE + offset1, mask, other=0).to(tl.float32)
        x3 = tl.load(TABLE + offset1 + 1, mask, other=0).to(tl.float32)
        t0, t1, t2, t3 = s & 3, (s >> 2) & 3, (s >> 4) & 3, (s >> 6) & 3
        a = tl.where(t0 == 1, x0, tl.where(t0 == 2, -x0, 0.0)) + tl.where(
            t2 == 1, x1, tl.where(t2 == 2, -x1, 0.0)
        )
        b = tl.where(t1 == 1, x2, tl.where(t1 == 2, -x2, 0.0)) + tl.where(
            t3 == 1, x3, tl.where(t3 == 2, -x3, 0.0)
        )
    else:
        a = tl.load(TABLE + base + p0 * 9 + q0, mask, other=0)
        b = tl.load(TABLE + base + p1 * 9 + q1, mask, other=0)
    value = tl.sum(a + b, 1) * tl.load(ALPHA + expert * N + ns, ns < N, other=0).to(
        tl.float32
    )
    value = value.to(tl.bfloat16).to(tl.float32)
    if FIRST:
        value = tl.maximum(value, 0)
        value *= value
    else:
        value *= tl.load(GATES + route)
    tl.store(Y + route * N + ns, value, ns < N)


def moe(
    x,
    up,
    down,
    alpha_up,
    alpha_down,
    meta_up,
    meta_down,
    ids,
    gates,
    block_n=16,
    direct=False,
):
    tokens, latent = x.shape
    experts, hidden, _ = up.shape
    top = ids.shape[1]
    h = torch.empty((tokens * top, hidden), device=x.device, dtype=x.dtype)
    partial = torch.empty((tokens * top, latent), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    for first, src, weight, alpha, meta, dst, n, k in (
        (True, x, up, alpha_up, meta_up, h, hidden, latent),
        (False, h, down, alpha_down, meta_down, partial, latent, hidden),
    ):
        if direct:
            table = src
        else:
            table = torch.empty(
                (src.shape[0], k // 2, 9), device=x.device, dtype=torch.float32
            )
            _tables[(triton.cdiv(table.numel(), 512),)](
                src, table, src.shape[0], k, 512, enable_fp_fusion=False
            )
        _apply[(tokens * top, triton.cdiv(n, block_n))](
            table,
            weight,
            meta,
            alpha,
            ids,
            gates,
            dst,
            n,
            k,
            top,
            first,
            block_n,
            triton.next_power_of_2(k // 8),
            DIRECT=direct,
            num_warps=4,
            enable_fp_fusion=False,
        )
    _combine[(tokens, triton.cdiv(latent, 128))](
        partial, y, latent, top, 128, triton.next_power_of_2(top)
    )
    return y
