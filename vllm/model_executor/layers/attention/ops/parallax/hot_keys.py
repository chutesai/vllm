# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded, direct-mapped decode key cache. Never owns historical model state.

Persistent latents remain authoritative. Writes invalidate the affected hot slot;
tags bind both physical latent chunk and logical RoPE position. Hash collisions
fall back to reconstruction and cannot remove selected tokens.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mark(
    SEL,
    POS,
    STARTS,
    TABLE,
    WANT,
    T: tl.constexpr,
    HK: tl.constexpr,
    TOP: tl.constexpr,
    NS: tl.constexpr,
    BS: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    CAP: tl.constexpr,
    BT: tl.constexpr,
):
    token, group = tl.program_id(0), tl.program_id(1)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < NS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), 0)
    pos = tl.load(POS + token)
    slots = tl.arange(0, BT)
    block = tl.load(SEL + (token * HK + group) * TOP + slots, slots < TOP, other=-1)
    chunks = tl.arange(0, 8)
    loc = block[:, None] * 128 + chunks[None, :] * 16
    valid = (
        (seq < NS)
        & (pos >= 0)
        & (slots[:, None] < TOP)
        & (block[:, None] >= 0)
        & (loc <= pos)
    )
    page = tl.load(TABLE + seq * TS + loc // PAGE, valid, other=0)
    physical = page.to(tl.int64) * (PAGE // 16) + (loc % PAGE) // 16
    tag = (physical << 32) | loc.to(tl.int64)
    tl.atomic_min(WANT + physical % CAP, tag, valid, sem="relaxed")


@triton.jit
def _project(
    C,
    WK,
    ROPE,
    WANT,
    TAGS,
    KEYS,
    CAP: tl.constexpr,
    PAGE: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    HK: tl.constexpr,
    MAXPOS: tl.constexpr,
):
    slot = tl.program_id(0)
    tag = tl.load(WANT + slot)
    old = tl.load(TAGS + slot)
    if tag != 9223372036854775807 and tag != old:
        physical = tag >> 32
        base = tag & 4294967295
        page = physical // (PAGE // 16)
        offset = (physical % (PAGE // 16)) * 16
        rows = tl.arange(0, 16)
        cols = tl.arange(0, 64)
        dd = tl.arange(0, HK * 128)
        acc = tl.full((16, HK * 128), 0, tl.float32)
        for part in range(4):
            cc = part * 64 + cols
            latent = tl.load(
                C + page * CS0 + (offset + rows[:, None]) * CS1 + cc[None, :]
            )
            weight = tl.load(WK + dd[None, :] * 256 + cc[:, None])
            acc = tl.dot(latent, weight, acc)
        k = acc.to(tl.bfloat16).to(tl.float32)
        partner = tl.gather(k, (dd ^ 1)[None, :] + tl.zeros((16, 1), tl.int32), 1)
        pos = base + rows
        cos = tl.load(
            ROPE + pos[:, None] * 128 + (dd[None, :] % 128) // 2,
            pos[:, None] < MAXPOS,
            other=0,
        ).to(tl.float32)
        sin = tl.load(
            ROPE + pos[:, None] * 128 + (dd[None, :] % 128) // 2 + 64,
            pos[:, None] < MAXPOS,
            other=0,
        ).to(tl.float32)
        k = (k * cos + tl.where(dd[None, :] % 2 == 0, -partner, partner) * sin).to(
            tl.bfloat16
        )
        tl.store(KEYS + (slot * 16 + rows[:, None]) * (HK * 128) + dd[None, :], k)
        tl.store(TAGS + slot, tag)


def prepare_hot_keys(q, latent, wk, rope, selected, positions, starts, table, hot):
    keys, tags = hot
    t, hk, top = selected.shape
    if hk != 4 or latent.shape[1] % 16:
        raise ValueError("Hot keys require four KV heads and pages divisible by 16")
    cap = tags.numel()
    wanted = torch.full_like(tags, torch.iinfo(torch.int64).max)
    ns = starts.numel() - 1
    _mark[(t, hk)](
        selected,
        positions,
        starts,
        table,
        wanted,
        t,
        hk,
        top,
        ns,
        triton.next_power_of_2(ns),
        table.stride(0),
        latent.shape[1],
        cap,
        triton.next_power_of_2(top),
    )
    _project[(cap,)](
        latent,
        wk,
        rope,
        wanted,
        tags,
        keys,
        cap,
        latent.shape[1],
        *latent.stride()[:2],
        hk,
        rope.shape[0],
        num_warps=8,
        num_stages=1,
        enable_fp_fusion=False,
    )
