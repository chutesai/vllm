# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Share KV loads across four query heads, retaining scalar FP32 reductions."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _statistics(q, k, maximum, valid):
    scores = tl.sum(k * q[None, :], axis=1) * (128**-0.5)
    scores = tl.where(valid, scores, -float("inf"))
    new_max = tl.maximum(maximum, tl.max(scores, axis=0))
    alpha = tl.exp(maximum - new_max)
    p = tl.exp(scores - new_max)
    return p, alpha, new_max


@triton.jit
def _accumulate(v, p, alpha, acc, denom):
    acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
    denom = denom * alpha + tl.sum(p, axis=0)
    return acc, denom


@triton.jit
def _grouped(
    Q,
    K,
    V,
    TABLE,
    STARTS,
    POS,
    SELECTED,
    OUT,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    TOP: tl.constexpr,
    NS: tl.int32,
    BS: tl.constexpr,
):
    token, group = tl.program_id(0), tl.program_id(1)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < NS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), axis=0)
    ds = tl.arange(0, 128)
    ns = tl.arange(0, 128)
    pos = tl.load(POS + token)
    if seq < NS and pos >= 0:
        q0 = tl.load(Q + (token * HQ + group * 4 + 0) * 128 + ds).to(tl.float32)
        acc0 = tl.full((128,), 0, tl.float32)
        m0, l0 = -float("inf"), 0.0
        q1 = tl.load(Q + (token * HQ + group * 4 + 1) * 128 + ds).to(tl.float32)
        acc1 = tl.full((128,), 0, tl.float32)
        m1, l1 = -float("inf"), 0.0
        q2 = tl.load(Q + (token * HQ + group * 4 + 2) * 128 + ds).to(tl.float32)
        acc2 = tl.full((128,), 0, tl.float32)
        m2, l2 = -float("inf"), 0.0
        q3 = tl.load(Q + (token * HQ + group * 4 + 3) * 128 + ds).to(tl.float32)
        acc3 = tl.full((128,), 0, tl.float32)
        m3, l3 = -float("inf"), 0.0
        for b in range(TOP):
            block = tl.load(SELECTED + (token * HK + group) * TOP + b)
            if block >= 0:
                loc = block * 128 + ns
                valid = loc <= pos
                page = tl.load(TABLE + seq * TS + loc // PAGE, valid, other=0)
                off = (
                    page[:, None] * K0
                    + (loc[:, None] % PAGE) * K1
                    + group * K2
                    + ds[None, :]
                )
                key = tl.load(K + off, valid[:, None], other=0).to(tl.float32)
                p0, a0, m0 = _statistics(q0, key, m0, valid)
                p1, a1, m1 = _statistics(q1, key, m1, valid)
                p2, a2, m2 = _statistics(q2, key, m2, valid)
                p3, a3, m3 = _statistics(q3, key, m3, valid)
                value = tl.load(V + off, valid[:, None], other=0).to(tl.float32)
                acc0, l0 = _accumulate(value, p0, a0, acc0, l0)
                acc1, l1 = _accumulate(value, p1, a1, acc1, l1)
                acc2, l2 = _accumulate(value, p2, a2, acc2, l2)
                acc3, l3 = _accumulate(value, p3, a3, acc3, l3)
        tl.store(OUT + (token * HQ + group * 4 + 0) * 128 + ds, acc0 / l0)
        tl.store(OUT + (token * HQ + group * 4 + 1) * 128 + ds, acc1 / l1)
        tl.store(OUT + (token * HQ + group * 4 + 2) * 128 + ds, acc2 / l2)
        tl.store(OUT + (token * HQ + group * 4 + 3) * 128 + ds, acc3 / l3)
    else:
        tl.store(OUT + (token * HQ + group * 4 + 0) * 128 + ds, 0.0)
        tl.store(OUT + (token * HQ + group * 4 + 1) * 128 + ds, 0.0)
        tl.store(OUT + (token * HQ + group * 4 + 2) * 128 + ds, 0.0)
        tl.store(OUT + (token * HQ + group * 4 + 3) * 128 + ds, 0.0)


def grouped_scalar_attention(
    q,
    k,
    v,
    selected,
    positions,
    starts,
    table,
    *,
    warps=8,
    maxnreg=None,
    return_kernel=False,
):
    t, hq, d = q.shape
    hk = k.shape[2]
    if d != 128 or hq != 4 * hk or k.stride() != v.stride() or k.stride(-1) != 1:
        raise ValueError("Four-query-head groups and matching 128D KV strides required")
    if warps not in (4, 8, 16, 32):
        raise ValueError("Unsupported grouped attention warp count")
    if maxnreg not in (None, 64, 96, 128, 160, 192):
        raise ValueError("Unsupported grouped attention register limit")
    out = torch.empty_like(q)
    ns = starts.numel() - 1
    launch = {} if maxnreg is None else {"maxnreg": maxnreg}
    kernel = _grouped[(t, hk)](
        q,
        k,
        v,
        table,
        starts,
        positions,
        selected,
        out,
        *k.stride()[:3],
        table.stride(0),
        k.shape[1],
        hq,
        hk,
        selected.shape[-1],
        ns,
        triton.next_power_of_2(ns),
        num_warps=warps,
        **launch,
    )
    return (out, kernel) if return_kernel else out
