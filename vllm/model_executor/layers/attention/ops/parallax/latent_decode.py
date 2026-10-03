# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Selected-block LMSA with persistent latents and on-chip K/V reconstruction.

K is rounded to BF16 before RoPE. V projection is absorbed after the
weighted latent sum: exact in real arithmetic; BF16 rounding differs.
Only the selected 128-token blocks are reconstructed. Historical latents and
index keys remain available for future queries selecting different blocks.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _latent_attend(
    Q,
    C,
    WK,
    WV,
    ROPE,
    SELECTED,
    POS,
    STARTS,
    TABLE,
    OUT,
    STATS,
    HOTKEYS,
    HOTTAGS,
    HOT: tl.constexpr,
    CAP: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    TOP: tl.constexpr,
    PAGE: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    TS: tl.constexpr,
    NS: tl.constexpr,
    BS: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    token, group = tl.program_id(0), tl.program_id(1)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < NS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), 0)
    pos = tl.load(POS + token)
    rr = tl.arange(0, 16)
    dd = tl.arange(0, 128)
    cc = tl.arange(0, 64)
    heads = group * (HQ // HK) + rr
    q = tl.load(
        Q + (token * HQ + heads[:, None]) * 128 + dd[None, :],
        (pos >= 0) & (rr[:, None] < HQ // HK),
        other=0,
    )
    acc = tl.full((16, 256), 0, tl.float32)
    denom = tl.full((16,), 0, tl.float32)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    if seq < NS and pos >= 0:
        for slot in range(tl.program_id(2), TOP, SPLITS):
            block = tl.load(SELECTED + (token * HK + group) * TOP + slot)
            if block >= 0:
                for chunk in range(128 // BN):
                    loc = block * 128 + chunk * BN + tl.arange(0, BN)
                    valid = loc <= pos
                    page = tl.load(TABLE + seq * TS + loc // PAGE, valid, other=0)
                    hit = tl.full((BN,), False, tl.int1)
                    physical = page.to(tl.int64) * (PAGE // 16) + (loc % PAGE) // 16
                    if HOT:
                        tag = (physical << 32) | (loc - loc % 16).to(tl.int64)
                        stored = tl.load(HOTTAGS + physical % CAP, valid, other=-1)
                        hit = valid & (tag == stored)
                    if HOT and tl.sum((valid & ~hit).to(tl.int32), 0) == 0:
                        key = tl.load(
                            HOTKEYS
                            + (
                                ((physical % CAP)[:, None] * 16 + (loc % 16)[:, None])
                                * HK
                                + group
                            )
                            * 128
                            + dd[None, :],
                            valid[:, None],
                            other=0,
                        )
                    else:
                        key_acc = tl.full((BN, 128), 0, tl.float32)
                        for piece in range(4):
                            col = piece * 64 + cc
                            latent = tl.load(
                                C
                                + page[:, None].to(tl.int64) * CS0
                                + (loc[:, None] % PAGE) * CS1
                                + col[None, :],
                                valid[:, None],
                                other=0,
                            )
                            wk = tl.load(
                                WK + (group * 128 + dd[None, :]) * 256 + col[:, None]
                            )
                            key_acc = tl.dot(latent, wk, key_acc)
                        key = key_acc.to(tl.bfloat16).to(tl.float32)
                        partner = tl.gather(
                            key, (dd ^ 1)[None, :] + tl.zeros((BN, 1), tl.int32), axis=1
                        )
                        cos = tl.load(
                            ROPE + loc[:, None] * 128 + (dd[None, :] // 2),
                            valid[:, None],
                            other=0,
                        ).to(tl.float32)
                        sin = tl.load(
                            ROPE + loc[:, None] * 128 + (dd[None, :] // 2) + 64,
                            valid[:, None],
                            other=0,
                        ).to(tl.float32)
                        key = (
                            key * cos
                            + tl.where(dd[None, :] % 2 == 0, -partner, partner) * sin
                        ).to(tl.bfloat16)
                    cd = tl.arange(0, 256)
                    value = tl.load(
                        C
                        + page[:, None].to(tl.int64) * CS0
                        + (loc[:, None] % PAGE) * CS1
                        + cd[None, :],
                        valid[:, None],
                        other=0,
                    )
                    scores = tl.dot(q, tl.trans(key)) * (128**-0.5)
                    scores = tl.where(valid[None, :], scores, -float("inf"))
                    new_max = tl.maximum(maximum, tl.max(scores, 1))
                    # A future tile in the local block must not poison the
                    # online softmax with exp(-inf - -inf).
                    if tl.min(loc, 0) <= pos:
                        alpha = tl.exp(maximum - new_max)
                        prob = tl.exp(scores - new_max[:, None])
                        acc = acc * alpha[:, None] + tl.dot(prob.to(tl.bfloat16), value)
                        denom = denom * alpha + tl.sum(prob, 1)
                        maximum = new_max
    partial_base = ((token * HK + group) * SPLITS + tl.program_id(2)) * (HQ // HK)
    tl.store(
        OUT + (partial_base + rr[:, None]) * 256 + tl.arange(0, 256)[None, :],
        acc,
        rr[:, None] < HQ // HK,
    )
    tl.store(STATS + (partial_base + rr) * 2, maximum, rr < HQ // HK)
    tl.store(STATS + (partial_base + rr) * 2 + 1, denom, rr < HQ // HK)


@triton.jit
def _combine(
    PART,
    STATS,
    WV,
    OUT,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    SPLITS: tl.constexpr,
    BT: tl.constexpr,
):
    token, group = tl.program_id(0), tl.program_id(1)
    slots = tl.arange(0, BT)
    rr = tl.arange(0, 16)
    cc = tl.arange(0, 64)
    dd = tl.arange(0, 128)
    base = ((token * HK + group) * SPLITS + slots[:, None]) * (HQ // HK) + rr[None, :]
    valid = (slots[:, None] < SPLITS) & (rr[None, :] < HQ // HK)
    maxima = tl.load(STATS + base * 2, valid, other=-float("inf"))
    denominators = tl.load(STATS + base * 2 + 1, valid, other=0)
    maximum = tl.max(maxima, 0)
    factors = tl.where(denominators > 0, tl.exp(maxima - maximum[None, :]), 0)
    denom = tl.maximum(tl.sum(denominators * factors, 0), 1e-20)
    output = tl.full((16, 128), 0, tl.float32)
    for piece in range(4):
        col = piece * 64 + cc
        parts = tl.load(
            PART + base[:, :, None] * 256 + col[None, None, :],
            valid[:, :, None],
            other=0,
        )
        mean = tl.sum(parts * factors[:, :, None], 0) / denom[:, None]
        weight = tl.load(WV + (group * 128 + dd[None, :]) * 256 + col[:, None]).to(
            tl.float32
        )
        output = tl.dot(mean, weight, output, input_precision="tf32x3")
    tl.store(
        OUT + ((token * HQ + group * (HQ // HK) + rr[:, None]) * 128 + dd[None, :]),
        output,
        rr[:, None] < HQ // HK,
    )


def latent_decode(
    q,
    latent,
    wk,
    wv,
    rope,
    selected,
    positions,
    starts,
    table,
    *,
    block_tokens=64,
    splits=None,
    num_warps=8,
    hot=None,
):
    t, hq, d = q.shape
    hk = selected.shape[1]
    if d != 128 or hq % hk or hq // hk > 16:
        raise ValueError("Expected 128-dimensional GQA with <=16 queries per KV head")
    if latent.shape[2:] != (1, 256) or latent.stride(-1) != 1:
        raise ValueError("Expected paged 256-dimensional latent cache")
    if wk.shape != (hk * 128, 256) or wv.shape != wk.shape:
        raise ValueError("Unexpected latent up-projection geometry")
    if any(x.dtype != torch.bfloat16 for x in (q, latent, wk, wv, rope)):
        raise ValueError("This prototype preserves BF16 projection and RoPE operands")
    if not all(x.is_contiguous() for x in (q, wk, wv, selected, rope)):
        raise ValueError("Expected contiguous projection weights, queries, and RoPE")
    if block_tokens not in (32, 64, 128):
        raise ValueError("Unsupported tile")
    out = torch.empty_like(q)
    ns = starts.numel() - 1
    top = selected.shape[-1]
    splits = min(top, 16 if t < 32 else 4) if splits is None else min(top, splits)
    partial = torch.empty(
        (t, hk, splits, hq // hk, 256), device=q.device, dtype=torch.float32
    )
    stats = torch.empty(
        (t, hk, splits, hq // hk, 2), device=q.device, dtype=torch.float32
    )
    _latent_attend[(t, hk, splits)](
        q,
        latent,
        wk,
        wv,
        rope,
        selected,
        positions,
        starts,
        table,
        partial,
        stats,
        hot[0] if hot is not None else latent,
        hot[1] if hot is not None else latent,
        hot is not None,
        hot[1].numel() if hot is not None else 1,
        hq,
        hk,
        selected.shape[-1],
        latent.shape[1],
        *latent.stride()[:2],
        table.stride(0),
        ns,
        triton.next_power_of_2(ns),
        block_tokens,
        splits,
        num_warps=num_warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    _combine[(t, hk)](
        partial,
        stats,
        wv,
        out,
        hq,
        hk,
        splits,
        triton.next_power_of_2(splits),
        num_warps=8,
        num_stages=1,
    )
    return out
