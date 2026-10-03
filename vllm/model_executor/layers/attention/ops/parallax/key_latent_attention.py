# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Projected keys plus shared latent values; no historical V expansion."""

import torch

from vllm.triton_utils import tl, triton

from .latent_decode import _combine


@triton.jit
def _attend(
    Q,
    K,
    C,
    SEL,
    POS,
    STARTS,
    TABLE,
    PART,
    STATS,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    TOP: tl.constexpr,
    PAGE: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    TS: tl.constexpr,
    NS: tl.constexpr,
    BS: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
):
    token, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < NS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), 0)
    pos = tl.load(POS + token)
    rr = tl.arange(0, 16)
    dd = tl.arange(0, 128)
    cc = tl.arange(0, 256)
    heads = group * (HQ // HK) + rr
    q = tl.load(
        Q + (token * HQ + heads[:, None]) * 128 + dd[None, :],
        (pos >= 0) & (rr[:, None] < HQ // HK),
        other=0,
    )
    acc = tl.full((16, 256), 0, tl.float32)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denom = tl.full((16,), 0, tl.float32)
    if seq < NS and pos >= 0:
        for slot in range(split, TOP, SPLITS):
            block = tl.load(SEL + (token * HK + group) * TOP + slot)
            if block >= 0:
                for chunk in range(128 // BN):
                    loc = block * 128 + chunk * BN + tl.arange(0, BN)
                    valid = loc <= pos
                    if tl.min(loc, 0) <= pos:
                        page = tl.load(
                            TABLE + seq * TS + loc // PAGE, valid, other=0
                        ).to(tl.int64)
                        k = tl.load(
                            K
                            + page[:, None] * K0
                            + (loc[:, None] % PAGE) * K1
                            + group * K2
                            + dd[None, :],
                            valid[:, None],
                            other=0,
                        )
                        latent = tl.load(
                            C
                            + page[:, None] * C0
                            + (loc[:, None] % PAGE) * C1
                            + cc[None, :],
                            valid[:, None],
                            other=0,
                        )
                        score = tl.dot(q, tl.trans(k)) * (128**-0.5)
                        score = tl.where(valid[None, :], score, -float("inf"))
                        nextmax = tl.maximum(maximum, tl.max(score, 1))
                        alpha = tl.exp(maximum - nextmax)
                        prob = tl.exp(score - nextmax[:, None])
                        acc = acc * alpha[:, None] + tl.dot(
                            prob.to(tl.bfloat16), latent
                        )
                        denom = denom * alpha + tl.sum(prob, 1)
                        maximum = nextmax
    base = ((token * HK + group) * SPLITS + split) * (HQ // HK)
    tl.store(
        PART + (base + rr[:, None]) * 256 + cc[None, :], acc, rr[:, None] < HQ // HK
    )
    tl.store(STATS + (base + rr) * 2, maximum, rr < HQ // HK)
    tl.store(STATS + (base + rr) * 2 + 1, denom, rr < HQ // HK)


def key_latent_attention(
    q, keys, latent, wv, selected, positions, starts, table, *, prefill=False
):
    t, hq, d = q.shape
    hk = selected.shape[1]
    top = selected.shape[-1]
    if d != 128 or hk != 4 or latent.shape[2:] != (1, 256):
        raise ValueError("Expected production LMSA geometry")
    splits = min(top, 1 if prefill else 8 if t < 32 else 4)
    part = torch.empty(
        t, hk, splits, hq // hk, 256, device=q.device, dtype=torch.float32
    )
    stats = torch.empty(
        t, hk, splits, hq // hk, 2, device=q.device, dtype=torch.float32
    )
    out = torch.empty_like(q)
    ns = starts.numel() - 1
    _attend[(t, hk, splits)](
        q,
        keys,
        latent,
        selected,
        positions,
        starts,
        table,
        part,
        stats,
        hq,
        hk,
        top,
        latent.shape[1],
        *keys.stride()[:3],
        *latent.stride()[:2],
        table.stride(0),
        ns,
        triton.next_power_of_2(ns),
        64,
        splits,
        num_warps=4,
        num_stages=1,
    )
    _combine[(t, hk)](
        part,
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


@triton.jit
def _store(
    K,
    C,
    INDEX_INPUT,
    SLOTS,
    ISLOTS,
    CACHE,
    INDEX,
    N: tl.constexpr,
    NS: tl.constexpr,
    NI: tl.constexpr,
    KS: tl.constexpr,
    CS: tl.constexpr,
    IS: tl.constexpr,
    PAGE: tl.constexpr,
    IPAGE: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, 1024)
    slot = tl.load(SLOTS + row, row < NS, other=-1)
    islot = tl.load(ISLOTS + row, row < NI, other=-1)
    k = tl.load(K + row * KS + d, (d < 512) & (slot >= 0), other=0)
    c = tl.load(C + row * CS + d - 512, (d >= 512) & (d < 768) & (slot >= 0), other=0)
    tl.store(
        CACHE + (slot // PAGE) * C0 + (slot % PAGE) * C1 + d,
        tl.where(d < 512, k, c),
        (d < 768) & (slot >= 0),
    )
    i = tl.load(INDEX_INPUT + row * IS + d, (d < 128) & (islot >= 0), other=0)
    tl.store(
        INDEX + (islot // IPAGE) * I0 + (islot % IPAGE) * I1 + d,
        i,
        (d < 128) & (islot >= 0),
    )


def write_key_latent_cache(keys, latent, index, slots, islots, cache, icache):
    if keys.shape[-1] != 512 or latent.shape[-1] != 256 or index.shape[-1] != 128:
        raise ValueError("Unexpected key/latent geometry")
    _store[(keys.shape[0],)](
        keys,
        latent,
        index,
        slots,
        islots,
        cache,
        icache,
        keys.shape[0],
        slots.numel(),
        islots.numel(),
        keys.stride(0),
        latent.stride(0),
        index.stride(0),
        cache.shape[1],
        icache.shape[1],
        *cache.stride()[:2],
        *icache.stride()[:2],
        num_warps=4,
    )
