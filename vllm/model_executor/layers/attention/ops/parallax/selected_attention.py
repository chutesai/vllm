# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Present selected LMSA blocks as virtual paged sequences, without copying KV."""

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


@triton.jit
def _tables(
    SELECTED,
    POS,
    STARTS,
    TABLE,
    VTABLE,
    LENS,
    CUSEQ,
    T: tl.int32,
    HK: tl.constexpr,
    TOP: tl.constexpr,
    PAGE: tl.constexpr,
    VPAGE: tl.constexpr,
    TS: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    UNIT: tl.constexpr,
    SEQS: tl.int32,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BP: tl.constexpr,
):
    token, group = tl.program_id(0), tl.program_id(1)
    virtual = token * HK + group
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < SEQS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), 0)
    pos = tl.load(POS + token)
    slots = tl.arange(0, BT)
    chosen = tl.load(
        SELECTED + (token * HK + group) * TOP + slots, slots < TOP, other=-1
    )
    count = tl.sum(((chosen >= 0) & (slots < TOP)).to(tl.int32), 0)
    current = pos // 128
    last = tl.sum(tl.where(slots == count - 1, chosen, 0), 0)
    ordered = tl.where(chosen == current, last, chosen)
    ordered = tl.where(slots == count - 1, current, ordered)
    # The partial current block must be last so ordinary causal masking works.
    columns = tl.arange(0, BP)
    ppb = 128 // VPAGE
    block = tl.gather(ordered, tl.minimum(columns // ppb, BT - 1), 0)
    loc = block * 128 + (columns % ppb) * VPAGE
    valid = (
        (seq < SEQS)
        & (pos >= 0)
        & (columns < count * ppb)
        & (columns < TOP * ppb)
        & (loc <= pos)
    )
    physical = tl.load(TABLE + seq * TS + loc // PAGE, valid, other=0)
    tl.store(
        VTABLE + virtual * (TOP * ppb) + columns,
        tl.where(
            valid,
            (physical.to(tl.int64) * KS0 + group * KS2 + (loc % PAGE) * KS1) // UNIT,
            0,
        ),
        columns < TOP * ppb,
    )
    length = tl.where((pos >= 0) & (count > 0), (count - 1) * 128 + pos % 128 + 1, 0)
    tl.store(LENS + virtual, length)
    tl.store(CUSEQ + virtual, virtual)
    if virtual == T * HK - 1:
        tl.store(CUSEQ + virtual + 1, virtual + 1)


@triton.jit
def _clear_empty(OUT, LENS, N: tl.int32, WIDTH: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(LENS + row) == 0:
        cols = tl.arange(0, B)
        tl.store(OUT + row * WIDTH + cols, 0, cols < WIDTH)


def selected_attention(q, k, v, selected, positions, starts, table):
    t, hq, d = q.shape
    hk = k.shape[2]
    page = k.shape[1]
    top = selected.shape[-1]
    vpage = math.gcd(page, 128)
    if k.stride(-1) != 1 or k.stride() != v.stride():
        raise ValueError(
            "Selected attention needs matching K/V strides, contiguous head dimension"
        )
    # Virtual page IDs are aligned element offsets. Their stride also
    # handles padding between physical pages in vLLM's hybrid cache. Views
    # overlap by design; only table-selected offsets are read, never written.
    last_base = (
        (k.shape[0] - 1) * k.stride(0)
        + (hk - 1) * k.stride(2)
        + (page - vpage) * k.stride(1)
    )
    unit = math.gcd(*k.stride()[:3])
    shape = (last_base // unit + 1, vpage, 1, d)
    stride = (unit, k.stride(1), unit, 1)
    vk = k.as_strided(shape, stride)
    vv = v.as_strided(shape, stride)
    seqs = starts.numel() - 1
    virtual_table = torch.empty(
        (t * hk, top * 128 // vpage), device=q.device, dtype=torch.int64
    )
    lengths = torch.empty(t * hk, device=q.device, dtype=torch.int32)
    cu = torch.empty(t * hk + 1, device=q.device, dtype=torch.int32)
    _tables[(t, hk)](
        selected,
        positions,
        starts,
        table,
        virtual_table,
        lengths,
        cu,
        t,
        hk,
        top,
        page,
        vpage,
        table.stride(0),
        *k.stride()[:3],
        unit,
        seqs,
        triton.next_power_of_2(top),
        triton.next_power_of_2(seqs),
        triton.next_power_of_2(top * 128 // vpage),
        num_warps=4,
    )
    out = torch.empty_like(q)
    unified_attention(
        q=q.view(t * hk, hq // hk, d),
        k=vk,
        v=vv,
        out=out.view(t * hk, hq // hk, d),
        cu_seqlens_q=cu,
        max_seqlen_q=1,
        seqused_k=lengths,
        max_seqlen_k=top * 128,
        softmax_scale=d**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=virtual_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    _clear_empty[(t * hk,)](
        out, lengths, t * hk, hq // hk * d, triton.next_power_of_2(hq // hk * d)
    )
    return out
