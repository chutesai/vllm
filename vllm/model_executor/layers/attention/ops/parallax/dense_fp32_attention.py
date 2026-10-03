# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""All-selected causal prefill: tiled BF16 QK and FP32 probability/value MMA.

This is an opt-in numerical candidate, not a bit-exact replacement for the
scalar selected-block recurrence. All keys must fit in the selected budget.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _dense(
    Q,
    K,
    V,
    STARTS,
    LENGTHS,
    TABLE,
    OUT,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    PAGE: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    TS: tl.constexpr,
    CONTEXT: tl.constexpr,
    BQ: tl.constexpr,
):
    seq, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(STARTS + seq)
    end = tl.load(STARTS + seq + 1)
    length = tl.load(LENGTHS + seq)
    rr = tile * BQ + tl.arange(0, BQ)
    dd = tl.arange(0, 128)
    nn = tl.arange(0, 128)
    live = start + rr < end
    pos = length - (end - start) + rr
    if tile * BQ < end - start:
        q = tl.load(
            Q + ((start + rr[:, None]) * HQ + head) * 128 + dd[None, :],
            live[:, None],
            other=0,
        )
        maximum = tl.full((BQ,), -float("inf"), tl.float32)
        denom = tl.full((BQ,), 0, tl.float32)
        acc = tl.full((BQ, 128), 0, tl.float32)
        for block in range(triton.cdiv(CONTEXT, 128)):
            loc = block * 128 + nn
            if block * 128 < length and block * 128 <= tl.max(
                tl.where(live, pos, -1), 0
            ):
                valid = loc < length
                page = tl.load(TABLE + seq * TS + loc // PAGE, valid, other=0)
                base = (
                    page.to(tl.int64) * K0
                    + (loc % PAGE) * K1
                    + (head // (HQ // HK)) * K2
                )
                key = tl.load(K + base[None, :] + dd[:, None], valid[None, :], other=0)
                scores = tl.dot(q, key) * (128**-0.5)
                mask = valid[None, :] & (loc[None, :] <= pos[:, None]) & live[:, None]
                scores = tl.where(mask, scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, 1))
                alpha = tl.exp(maximum - new_max)
                p = tl.exp(scores - new_max[:, None])
                p = tl.where(mask, p, 0.0)
                value = tl.load(
                    V + base[:, None] + dd[None, :], valid[:, None], other=0
                ).to(tl.float32)
                acc = acc * alpha[:, None] + tl.dot(p, value, input_precision="tf32x3")
                denom = denom * alpha + tl.sum(p, 1)
                maximum = new_max
        tl.store(
            OUT + ((start + rr[:, None]) * HQ + head) * 128 + dd[None, :],
            acc / denom[:, None],
            live[:, None],
        )


def dense_fp32_attention(
    q, k, v, starts, lengths, table, max_query, context, *, query_tile=32
):
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("BF16 stored operands required")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or k.stride() != v.stride():
        raise ValueError("Matching strided 128D KV required")
    if context > 2048 or query_tile not in (16, 32, 64):
        raise ValueError("Only all-selected short-context prefill is supported")
    out = torch.empty_like(q)
    _dense[(starts.numel() - 1, q.shape[1], triton.cdiv(max_query, query_tile))](
        q,
        k,
        v,
        starts,
        lengths,
        table,
        out,
        q.shape[1],
        k.shape[2],
        k.shape[1],
        *k.stride()[:3],
        table.stride(0),
        context,
        query_tile,
        num_warps=4,
    )
    return out
