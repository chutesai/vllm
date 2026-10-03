# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Split selected-block decode for the expanded-KV throughput profile."""

import torch

from vllm.triton_utils import tl, triton


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
    cc = tl.arange(0, 128)
    heads = group * (HQ // HK) + rr
    q = tl.load(
        Q + (token * HQ + heads[:, None]) * 128 + dd[None, :],
        (pos >= 0) & (rr[:, None] < HQ // HK),
        other=0,
    )
    acc = tl.full((16, 128), 0, tl.float32)
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
                            + group * K2
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
        PART + (base + rr[:, None]) * 128 + cc[None, :], acc, rr[:, None] < HQ // HK
    )
    tl.store(STATS + (base + rr) * 2, maximum, rr < HQ // HK)
    tl.store(STATS + (base + rr) * 2 + 1, denom, rr < HQ // HK)


@triton.jit
def _combine(
    PART,
    STATS,
    OUT,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    SPLITS: tl.constexpr,
    BT: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    group = head // (HQ // HK)
    h = head % (HQ // HK)
    ss = tl.arange(0, BT)
    dd = tl.arange(0, 128)
    base = ((token * HK + group) * SPLITS + ss) * (HQ // HK) + h
    maxima = tl.load(STATS + base * 2, ss < SPLITS, other=-float("inf"))
    den = tl.load(STATS + base * 2 + 1, ss < SPLITS, other=0)
    m = tl.max(maxima, 0)
    factor = tl.where(den > 0, tl.exp(maxima - m), 0)
    values = tl.load(
        PART + base[:, None] * 128 + dd[None, :], ss[:, None] < SPLITS, other=0
    )
    result = tl.sum(values * factor[:, None], 0) / tl.maximum(
        tl.sum(den * factor, 0), 1e-20
    )
    tl.store(OUT + (token * HQ + head) * 128 + dd, result)


def split_selected_attention(q, keys, values, selected, positions, starts, table):
    t, hq, d = q.shape
    hk = selected.shape[1]
    top = selected.shape[-1]
    if d != 128 or hq % hk or hq // hk > 16 or keys.stride() != values.stride():
        raise ValueError("Unexpected grouped attention geometry")
    splits = min(top, 16 if t < 32 else 4)
    part = torch.empty(
        t, hk, splits, hq // hk, 128, device=q.device, dtype=torch.float32
    )
    stats = torch.empty(
        t, hk, splits, hq // hk, 2, device=q.device, dtype=torch.float32
    )
    out = torch.empty_like(q)
    ns = starts.numel() - 1
    _attend[(t, hk, splits)](
        q,
        keys,
        values,
        selected,
        positions,
        starts,
        table,
        part,
        stats,
        hq,
        hk,
        top,
        keys.shape[1],
        *keys.stride()[:3],
        *values.stride()[:2],
        table.stride(0),
        ns,
        triton.next_power_of_2(ns),
        64,
        splits,
        num_warps=4,
        num_stages=1,
    )
    _combine[(t, hq)](
        part, stats, out, hq, hk, splits, triton.next_power_of_2(splits), num_warps=4
    )
    return out
