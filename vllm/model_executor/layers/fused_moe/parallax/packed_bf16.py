# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Losslessly packed ternary weights, BF16 operands and FP32 MMA accumulation."""

import torch

from vllm.triton_utils import tl, triton

from ..moe_align_block_size import (
    moe_align_block_size,
)
from .ternary_decode import _combine


@triton.jit
def _grouped(
    X,
    W,
    A,
    GATES,
    SORTED,
    EXPERTS,
    COUNT,
    Y,
    ROUTES: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOP: tl.constexpr,
    FIRST: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PACKED: tl.constexpr,
    UNIQUE_DECODE: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    block, column = tl.program_id(0), tl.program_id(1)
    if block * BM < tl.load(COUNT):
        expert = tl.load(EXPERTS + block)
        routes = tl.load(SORTED + block * BM + tl.arange(0, BM))
        rows = routes // TOP if FIRST else routes
        ns = column * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            kk = start * BK + ks
            x = tl.load(
                X + rows[:, None] * K + kk[None, :],
                (routes[:, None] < ROUTES) & (kk[None, :] < K),
                other=0,
            )
            if PACKED:
                if UNIQUE_DECODE:
                    byte_k = start * (BK // 4) + tl.arange(0, BK // 4)
                    if TRANSPOSED:
                        addresses = (
                            W + (expert * (K // 4) + byte_k[:, None]) * N + ns[None, :]
                        )
                    else:
                        addresses = (
                            W + (expert * N + ns[None, :]) * (K // 4) + byte_k[:, None]
                        )
                    packed = tl.load(
                        addresses,
                        (ns[None, :] < N) & (byte_k[:, None] < K // 4),
                        other=0,
                    )
                    shift = tl.arange(0, 4) * 2
                    code = ((packed[:, None, :] >> shift[None, :, None]) & 3).reshape(
                        BK, BN
                    )
                else:
                    packed = tl.load(
                        W + (expert * N + ns[None, :]) * (K // 4) + kk[:, None] // 4,
                        (ns[None, :] < N) & (kk[:, None] < K),
                        other=0,
                    )
                    code = (packed >> ((kk[:, None] % 4) * 2)) & 3
                scale = tl.load(A + expert * N + ns, ns < N, other=0).to(tl.float32)
                weight = tl.where(
                    code == 1, scale[None, :], tl.where(code == 2, -scale[None, :], 0.0)
                )
            else:
                weight = tl.load(
                    W + (expert * N + ns[None, :]) * K + kk[:, None],
                    (ns[None, :] < N) & (kk[:, None] < K),
                    other=0,
                )
            acc = tl.dot(x, weight.to(x.dtype), acc)
        acc = acc.to(tl.bfloat16).to(tl.float32)
        if FIRST:
            acc = tl.maximum(acc, 0.0)
            acc *= acc
        else:
            acc *= tl.load(GATES + routes, routes < ROUTES, other=0)[:, None]
        tl.store(
            Y + routes[:, None] * N + ns[None, :],
            acc,
            (routes[:, None] < ROUTES) & (ns[None, :] < N),
        )


def moe(
    x,
    up,
    down,
    alpha_up,
    alpha_down,
    ids,
    gates,
    block_m=16,
    *,
    packed=True,
    block_n=64,
    block_k=64,
    unique_decode=False,
    transposed=False,
):
    tokens, latent = x.shape
    experts = up.shape[0]
    hidden = up.shape[2] if transposed else up.shape[1]
    top = ids.shape[1]
    if (
        x.dtype != torch.bfloat16
        or up.dtype != (torch.uint8 if packed else torch.bfloat16)
        or down.dtype != (torch.uint8 if packed else torch.bfloat16)
        or latent % 4
        or hidden % 4
        or down.shape
        != (
            (experts, hidden // 4, latent)
            if transposed
            else (experts, latent, hidden // 4 if packed else hidden)
        )
        or up.shape
        != (
            (experts, latent // 4, hidden)
            if transposed
            else (experts, hidden, latent // 4 if packed else latent)
        )
        or (transposed and (not packed or not unique_decode))
        or (packed and alpha_up.shape != (experts, hidden))
        or (packed and alpha_down.shape != (experts, latent))
        or ids.shape != gates.shape
        or ids.shape[0] != tokens
    ):
        raise ValueError("Grouped ternary BF16 input/routing/weight shape mismatch")
    if any(
        not a.is_cuda or not a.is_contiguous() or a.device != x.device
        for a in (x, up, down, alpha_up, alpha_down, ids, gates)
    ):
        raise ValueError("Contiguous tensors on one CUDA device required")
    sorted_ids, expert_ids, count = moe_align_block_size(
        ids, block_m, experts, pad_sorted_ids=True
    )
    h = torch.empty((tokens * top, hidden), device=x.device, dtype=x.dtype)
    partial = torch.empty((tokens * top, latent), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    for first, src, weight, alpha, dst, n, k in (
        (True, x, up, alpha_up, h, hidden, latent),
        (False, h, down, alpha_down, partial, latent, hidden),
    ):
        _grouped[(expert_ids.numel(), triton.cdiv(n, block_n))](
            src,
            weight,
            alpha,
            gates,
            sorted_ids,
            expert_ids,
            count,
            dst,
            tokens * top,
            n,
            k,
            top,
            first,
            block_m,
            block_n,
            block_k,
            PACKED=packed,
            UNIQUE_DECODE=unique_decode,
            TRANSPOSED=transposed,
            num_warps=4,
        )
    _combine[(tokens, triton.cdiv(latent, 128))](
        partial, y, latent, top, 128, triton.next_power_of_2(top)
    )
    return y
