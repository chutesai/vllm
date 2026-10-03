# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Delay sparse-attention value loads without changing reduction arithmetic."""

from vllm.triton_utils import tl, triton


@triton.jit
def _attend_late_v(
    Q,
    K,
    V,
    TABLE,
    STARTS,
    LENS,
    SELECTED,
    OUT,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    TOP: tl.constexpr,
    SEQS: tl.int32,
    BS: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    group = head // (HQ // HK)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < SEQS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), axis=0)
    if seq < SEQS:
        begin, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
        pos = tl.load(LENS + seq) - (end - begin) + token - begin
        ds = tl.arange(0, 128)
        ns = tl.arange(0, 128)
        q = tl.load(Q + (token * HQ + head) * 128 + ds).to(tl.float32)
        acc = tl.full((128,), 0, tl.float32)
        maximum, denom = -float("inf"), 0.0
        for b in range(TOP):
            block = tl.load(SELECTED + (token * HK + group) * TOP + b)
            if block >= 0:
                loc = block * 128 + ns
                page = tl.load(TABLE + seq * TS + loc // PAGE, loc <= pos, other=0)
                off = page[:, None] * KS0 + (loc[:, None] % PAGE) * KS1
                off += group * KS2 + ds[None, :]
                k = tl.load(K + off, loc[:, None] <= pos, other=0).to(tl.float32)
                scores = tl.sum(k * q[None, :], axis=1) * (128**-0.5)
                scores = tl.where(loc <= pos, scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, axis=0))
                alpha = tl.exp(maximum - new_max)
                p = tl.exp(scores - new_max)
                v = tl.load(V + off, loc[:, None] <= pos, other=0).to(tl.float32)
                acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
                denom = denom * alpha + tl.sum(p, axis=0)
                maximum = new_max
        tl.store(OUT + (token * HQ + head) * 128 + ds, acc / denom)
    else:
        ds = tl.arange(0, 128)
        tl.store(OUT + (token * HQ + head) * 128 + ds, 0.0)
