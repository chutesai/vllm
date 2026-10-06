# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lambda paged attention with FP32 scores, probabilities and accumulation."""

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
    D: tl.constexpr,
    PAGE: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    TS: tl.constexpr,
    CONTEXT: tl.constexpr,
    WINDOW: tl.constexpr,
    BQ: tl.constexpr,
):
    seq, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
    length = tl.load(LENGTHS + seq)
    rr = tile * BQ + tl.arange(0, BQ)
    dd = tl.arange(0, D)
    nn = tl.arange(0, 128)
    live = start + rr < end
    pos = length - (end - start) + rr
    if tile * BQ < end - start:
        q = tl.load(
            Q + ((start + rr[:, None]) * HQ + head) * D + dd[None, :],
            live[:, None],
            other=0,
        ).to(tl.float32)
        maximum = tl.full((BQ,), -float("inf"), tl.float32)
        denom = tl.full((BQ,), 0, tl.float32)
        acc = tl.full((BQ, D), 0, tl.float32)
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
                key = tl.load(
                    K + base[None, :] + dd[:, None], valid[None, :], other=0
                ).to(tl.float32)
                scores = tl.dot(q, key, input_precision="tf32x3") * (D**-0.5)
                mask = valid[None, :] & (loc[None, :] <= pos[:, None]) & live[:, None]
                if WINDOW:
                    mask = mask & (loc[None, :] > pos[:, None] - WINDOW)
                scores = tl.where(mask, scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, 1))
                safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
                alpha = tl.exp(maximum - safe_max)
                p = tl.where(mask, tl.exp(scores - safe_max[:, None]), 0.0)
                value = tl.load(
                    V + base[:, None] + dd[None, :], valid[:, None], other=0
                ).to(tl.float32)
                acc = acc * alpha[:, None] + tl.dot(p, value, input_precision="tf32x3")
                denom = denom * alpha + tl.sum(p, 1)
                maximum = new_max
        tl.store(
            OUT + ((start + rr[:, None]) * HQ + head) * D + dd[None, :],
            acc / tl.maximum(denom[:, None], 1e-20),
            live[:, None],
        )


def dense_attention(q, k, v, metadata, window, context):
    out = torch.zeros_like(q)
    _dense[
        (
            metadata.query_start_loc.numel() - 1,
            q.shape[1],
            triton.cdiv(metadata.max_query_len, 16),
        )
    ](
        q,
        k,
        v,
        metadata.query_start_loc,
        metadata.seq_lens,
        metadata.block_table,
        out,
        q.shape[1],
        k.shape[2],
        q.shape[-1],
        k.shape[1],
        *k.stride()[:3],
        metadata.block_table.stride(0),
        context,
        window or 0,
        16,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
