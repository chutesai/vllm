# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tensor-core attention directly over selected LMSA blocks and physical KV pages."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _attend_selected(
    Q,
    K,
    V,
    SELECTED,
    POS,
    STARTS,
    TABLE,
    OUT,
    T: tl.int32,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    TOP: tl.constexpr,
    PAGE: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    TS: tl.constexpr,
    NS: tl.int32,
    BS: tl.constexpr,
    BT: tl.constexpr,
    BQ: tl.constexpr,
    BN: tl.constexpr,
    BOUNDED: tl.constexpr,
    FP32_PV: tl.constexpr,
):
    token, group = tl.program_id(0), tl.program_id(1)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < NS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), 0)
    pos = tl.load(POS + token)
    rr = tl.arange(0, BQ)
    dd = tl.arange(0, 128)
    heads = group * (HQ // HK) + rr
    live = (seq < NS) & (pos >= 0)
    q = tl.load(
        Q + (token * HQ + heads[:, None]) * 128 + dd[None, :],
        live & (rr[:, None] < HQ // HK),
        other=0,
    )
    acc = tl.full((BQ, 128), 0, tl.float32)
    denom = tl.full((BQ,), 0, tl.float32)
    maximum = tl.full((BQ,), -float("inf"), tl.float32)
    if live:
        slots = tl.arange(0, BT)
        chosen = tl.load(
            SELECTED + (token * HK + group) * TOP + slots, slots < TOP, other=-1
        )
        count = tl.sum(((slots < TOP) & (chosen >= 0)).to(tl.int32), 0)
        current = pos // 128
        last = tl.sum(tl.where(slots == count - 1, chosen, 0), 0)
        ordered = tl.where(chosen == current, last, chosen)
        ordered = tl.where(slots == count - 1, current, ordered)
        if FP32_PV:
            # Match the original selected-block order. Per-key causal masks
            # already handle the partial current block in this kernel.
            ordered = chosen
        limit = count * (128 // BN) if BOUNDED else TOP * (128 // BN)
        for i in range(limit):
            block = tl.sum(tl.where(slots == i // (128 // BN), ordered, 0), 0)
            loc = block * 128 + i % (128 // BN) * BN + tl.arange(0, BN)
            if i // (128 // BN) < count and block >= 0 and tl.min(loc, 0) <= pos:
                valid = loc <= pos
                page = tl.load(TABLE + seq * TS + loc // PAGE, valid, other=0)
                base = page.to(tl.int64) * KS0 + (loc % PAGE) * KS1 + group * KS2
                key = tl.load(K + base[None, :] + dd[:, None], valid[None, :], other=0)
                if FP32_PV:
                    scores = tl.dot(
                        q.to(tl.float32), key.to(tl.float32), input_precision="tf32x3"
                    ) * (128**-0.5)
                else:
                    scores = tl.dot(q, key) * (128**-0.5)
                scores = tl.where(valid[None, :], scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, 1))
                alpha = tl.exp(maximum - new_max)
                prob = tl.exp(scores - new_max[:, None])
                value = tl.load(
                    V + base[:, None] + dd[None, :], valid[:, None], other=0
                )
                if FP32_PV:
                    acc = acc * alpha[:, None] + tl.dot(
                        prob, value.to(tl.float32), input_precision="tf32x3"
                    )
                else:
                    acc = acc * alpha[:, None] + tl.dot(prob.to(value.dtype), value)
                denom = denom * alpha + tl.sum(prob, 1)
                maximum = new_max
    result = acc / tl.maximum(denom[:, None], 1e-20)
    tl.store(
        OUT + (token * HQ + heads[:, None]) * 128 + dd[None, :],
        result,
        rr[:, None] < HQ // HK,
    )


def direct_selected_attention(
    q,
    k,
    v,
    selected,
    positions,
    starts,
    table,
    *,
    block_tokens=128,
    warps=4,
    bounded_loop=False,
    fp32_probabilities=False,
):
    t, hq, d = q.shape
    hk = k.shape[2]
    if d != 128 or hq % hk or hq // hk > 16:
        raise ValueError(
            "128-dimensional grouped attention with at most 16 query heads"
        )
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("BF16 operands required")
    if k.stride() != v.stride() or k.stride(-1) != 1:
        raise ValueError("Matching KV strides with contiguous head dimension required")
    if not q.is_contiguous() or not selected.is_contiguous():
        raise ValueError("Contiguous queries and selected block IDs required")
    if block_tokens not in (32, 64, 128) or warps not in (4, 8):
        raise ValueError("Unsupported attention tile")
    out = torch.empty_like(q)
    ns = starts.numel() - 1
    _attend_selected[(t, hk)](
        q,
        k,
        v,
        selected,
        positions,
        starts,
        table,
        out,
        t,
        hq,
        hk,
        selected.shape[-1],
        k.shape[1],
        *k.stride()[:3],
        table.stride(0),
        ns,
        triton.next_power_of_2(ns),
        triton.next_power_of_2(selected.shape[-1]),
        16,
        block_tokens,
        bounded_loop,
        fp32_probabilities,
        num_warps=warps,
    )
    return out
