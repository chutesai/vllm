# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Selected-block LMSA with persistent latents and on-chip K/V reconstruction.

K/V are rounded to BF16 before RoPE/attention, as in the expanded reference.
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
    CACHE,
    MAP,
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
    NB: tl.constexpr,
    SPLITS: tl.constexpr,
    FALLBACK: tl.constexpr,
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
    acc = tl.full((16, 128), 0, tl.float32)
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
                    cached = tl.load(MAP + (seq * HK + group) * NB + block)
                    if not FALLBACK or cached >= 0:
                        row = chunk * BN + tl.arange(0, BN)
                        key = tl.load(
                            CACHE + (cached * 128 + row[:, None]) * 256 + dd[None, :]
                        )
                        value = tl.load(
                            CACHE
                            + (cached * 128 + row[:, None]) * 256
                            + 128
                            + dd[None, :]
                        )
                    else:
                        key_acc = tl.full((BN, 128), 0, tl.float32)
                        value_acc = tl.full((BN, 128), 0, tl.float32)
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
                            wv = tl.load(
                                WV + (group * 128 + dd[None, :]) * 256 + col[:, None]
                            )
                            key_acc = tl.dot(latent, wk, key_acc)
                            value_acc = tl.dot(latent, wv, value_acc)
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
                        value = value_acc.to(tl.bfloat16)
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
    tl.store(
        OUT + (token * HQ + heads[:, None]) * 128 + dd[None, :],
        acc / tl.maximum(denom[:, None], 1e-20),
        rr[:, None] < HQ // HK,
    )


def latent_cached_attention(
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
    block_tokens=128,
    max_context=None,
    capacity_blocks=4096,
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
    from .reconstructed_cache import (
        reconstruct_selected,
    )

    if max_context is None:
        max_context = rope.shape[0]
    cache, mapping, nb = reconstruct_selected(
        latent,
        wk,
        wv,
        rope,
        selected,
        positions,
        starts,
        table,
        max_context=max_context,
        capacity_blocks=capacity_blocks,
    )
    top = selected.shape[-1]
    fallback = ns * hk * nb > capacity_blocks
    _latent_attend[(t, hk, 1)](
        q,
        latent,
        wk,
        wv,
        rope,
        selected,
        positions,
        starts,
        table,
        out,
        out,
        cache,
        mapping,
        hq,
        hk,
        top,
        latent.shape[1],
        *latent.stride()[:2],
        table.stride(0),
        ns,
        triton.next_power_of_2(ns),
        block_tokens,
        nb,
        1,
        fallback,
        num_warps=8 if fallback else 4,
        num_stages=1,
        enable_fp_fusion=not fallback,
    )
    return out
