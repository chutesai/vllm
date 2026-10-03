# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded per-prefill reconstruction cache for the exact selected-block union.

The compact persistent cache remains unchanged. Blocks beyond the scratch budget
are explicitly mapped to -1 and reconstructed by the attention kernel instead.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mark(
    SELECTED,
    STARTS,
    FLAGS,
    T: tl.constexpr,
    HK: tl.constexpr,
    TOP: tl.constexpr,
    NB: tl.constexpr,
    NS: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
):
    token, group = tl.program_id(0), tl.program_id(1)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < NS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), 0)
    slots = tl.arange(0, BT)
    ids = tl.load(SELECTED + (token * HK + group) * TOP + slots, slots < TOP, other=-1)
    valid = (seq < NS) & (slots < TOP) & (ids >= 0) & (ids < NB)
    tl.atomic_or(FLAGS + (seq * HK + group) * NB + ids, 1, valid, sem="relaxed")


@triton.jit
def _project(
    C,
    WK,
    WV,
    ROPE,
    TABLE,
    STARTS,
    POS,
    MAP,
    CACHE,
    NB: tl.constexpr,
    HK: tl.constexpr,
    CAP: tl.constexpr,
    PAGE: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    TS: tl.constexpr,
):
    logical = tl.program_id(0)
    slot = tl.load(MAP + logical)
    if slot >= 0:
        block = logical % NB
        group = (logical // NB) % HK
        seq = logical // (NB * HK)
        end = tl.load(STARTS + seq + 1)
        begin = tl.load(STARTS + seq)
        pos = tl.load(POS + end - 1, end > begin, other=-1)
        col = tl.arange(0, 64)
        dd = tl.arange(0, 128)
        for half in range(2):
            loc = block * 128 + half * 64 + tl.arange(0, 64)
            valid = loc <= pos
            page = tl.load(TABLE + seq * TS + loc // PAGE, valid, other=0)
            ka = tl.full((64, 128), 0, tl.float32)
            va = tl.full((64, 128), 0, tl.float32)
            for part in range(4):
                cc = part * 64 + col
                latent = tl.load(
                    C
                    + page[:, None].to(tl.int64) * CS0
                    + (loc[:, None] % PAGE) * CS1
                    + cc[None, :],
                    valid[:, None],
                    other=0,
                )
                wk = tl.load(WK + (group * 128 + dd[None, :]) * 256 + cc[:, None])
                wv = tl.load(WV + (group * 128 + dd[None, :]) * 256 + cc[:, None])
                ka = tl.dot(latent, wk, ka)
                va = tl.dot(latent, wv, va)
            k = ka.to(tl.bfloat16).to(tl.float32)
            other = tl.gather(k, (dd[None, :] ^ 1) + tl.zeros((64, 1), tl.int32), 1)
            cos = tl.load(
                ROPE + loc[:, None] * 128 + (dd[None, :] // 2), valid[:, None], other=0
            ).to(tl.float32)
            sin = tl.load(
                ROPE + loc[:, None] * 128 + (dd[None, :] // 2) + 64,
                valid[:, None],
                other=0,
            ).to(tl.float32)
            k = (k * cos + tl.where(dd[None, :] % 2 == 0, -other, other) * sin).to(
                tl.bfloat16
            )
            row = half * 64 + tl.arange(0, 64)
            tl.store(CACHE + (slot * 128 + row[:, None]) * 256 + dd[None, :], k)
            tl.store(CACHE + (slot * 128 + row[:, None]) * 256 + 128 + dd[None, :], va)


def reconstruct_selected(
    latent,
    wk,
    wv,
    rope,
    selected,
    positions,
    starts,
    table,
    *,
    max_context,
    capacity_blocks=4096,
):
    t, hk, top = selected.shape
    ns = starts.numel() - 1
    nb = triton.cdiv(max_context, 128)
    flags = torch.zeros(ns * hk * nb, device=latent.device, dtype=torch.int32)
    _mark[(t, hk)](
        selected,
        starts,
        flags,
        t,
        hk,
        top,
        nb,
        ns,
        triton.next_power_of_2(ns),
        triton.next_power_of_2(top),
    )
    offsets = flags.cumsum(0, dtype=torch.int32) - 1
    cap = min(capacity_blocks, flags.numel())
    mapping = torch.where((flags != 0) & (offsets < cap), offsets, -1)
    cache = torch.empty((cap, 128, 256), device=latent.device, dtype=torch.bfloat16)
    _project[(flags.numel(),)](
        latent,
        wk,
        wv,
        rope,
        table,
        starts,
        positions,
        mapping,
        cache,
        nb,
        hk,
        cap,
        latent.shape[1],
        *latent.stride()[:2],
        table.stride(0),
        num_warps=8,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return cache, mapping, nb
