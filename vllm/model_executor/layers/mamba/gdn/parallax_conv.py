# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ragged causal depthwise convolution with vLLM-owned request state."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _conv(
    X,
    W,
    S,
    Y,
    STARTS,
    SLOTS,
    RESET,
    D: tl.constexpr,
    XS: tl.constexpr,
    SD0: tl.constexpr,
    SD1: tl.constexpr,
    SD2: tl.constexpr,
    ROUND_BEFORE_SILU: tl.constexpr,
    BD: tl.constexpr,
):
    seq = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    begin, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
    slot = tl.load(SLOTS + seq)
    if slot >= 0 and end > begin:
        reset = tl.load(RESET + seq)
        base = slot * SD0 + d * SD1
        s0 = tl.load(S + base, (d < D) & ~reset, other=0).to(tl.float32)
        s1 = tl.load(S + base + SD2, (d < D) & ~reset, other=0).to(tl.float32)
        s2 = tl.load(S + base + SD2 * 2, (d < D) & ~reset, other=0).to(tl.float32)
        w0 = tl.load(W + d * 4, d < D, other=0).to(tl.float32)
        w1 = tl.load(W + d * 4 + 1, d < D, other=0).to(tl.float32)
        w2 = tl.load(W + d * 4 + 2, d < D, other=0).to(tl.float32)
        w3 = tl.load(W + d * 4 + 3, d < D, other=0).to(tl.float32)
        for t in range(begin, end):
            x = tl.load(X + t * XS + d, d < D, other=0).to(tl.float32)
            y = s0 * w0 + s1 * w1 + s2 * w2 + x * w3
            if ROUND_BEFORE_SILU:
                y = y.to(Y.dtype.element_ty).to(tl.float32)
            tl.store(Y + t * D + d, y * tl.sigmoid(y), d < D)
            s0, s1, s2 = s1, s2, x
        tl.store(S + base, s0, d < D)
        tl.store(S + base + SD2, s1, d < D)
        tl.store(S + base + SD2 * 2, s2, d < D)


@triton.jit
def _parallel_conv(
    X,
    W,
    S,
    Y,
    STARTS,
    SLOTS,
    RESET,
    T: tl.int32,
    D: tl.constexpr,
    XS: tl.constexpr,
    SD0: tl.constexpr,
    SD1: tl.constexpr,
    SD2: tl.constexpr,
    ROUND_BEFORE_SILU: tl.constexpr,
    NS: tl.int32,
    BS: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < NS, other=2147483647)
    seq = tl.sum((t[:, None] >= ends[None, :]).to(tl.int32), 1)
    active = (t < T) & (seq < NS)
    begin = tl.load(STARTS + seq, active, other=0)
    slot = tl.load(SLOTS + seq, active, other=-1)
    reset = tl.load(RESET + seq, active, other=1)
    live = active & (slot >= 0)
    y = tl.full((BT, BD), 0, tl.float32)
    for j in tl.static_range(4):
        src = t - 3 + j
        fresh = src >= begin
        a = tl.load(
            X + src[:, None] * XS + d[None, :],
            (live & fresh)[:, None] & (d < D)[None, :],
            other=0,
        ).to(tl.float32)
        old = tl.load(
            S
            + slot[:, None] * SD0
            + d[None, :] * SD1
            + (src - begin + 3)[:, None] * SD2,
            (live & ~fresh & ~reset)[:, None] & (d < D)[None, :],
            other=0,
        ).to(tl.float32)
        w = tl.load(W + d * 4 + j, d < D, other=0).to(tl.float32)
        # Match the serial kernel's left-associated multiply/add expression.
        if j == 0:  # noqa: SIM108 -- constexpr preserves the first multiply
            y = (a + old) * w[None, :]
        else:
            y = y + (a + old) * w[None, :]
    if ROUND_BEFORE_SILU:
        y = y.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(
        Y + t[:, None] * D + d[None, :],
        y * tl.sigmoid(y),
        (t < T)[:, None] & (d < D)[None, :],
    )


@triton.jit
def _commit_conv(
    X,
    S,
    STARTS,
    SLOTS,
    RESET,
    D: tl.constexpr,
    XS: tl.constexpr,
    SD0: tl.constexpr,
    SD1: tl.constexpr,
    SD2: tl.constexpr,
    BD: tl.constexpr,
):
    seq = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    begin, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
    slot = tl.load(SLOTS + seq)
    if slot >= 0 and end > begin:
        reset = tl.load(RESET + seq)
        j = tl.arange(0, 4)
        src = end - 3 + j
        fresh = src >= begin
        value = tl.load(
            X + src[:, None] * XS + d[None, :],
            (j < 3)[:, None] & fresh[:, None] & (d < D)[None, :],
            other=0,
        )
        old = tl.load(
            S + slot * SD0 + d[None, :] * SD1 + (src - begin + 3)[:, None] * SD2,
            (j < 3)[:, None] & ~fresh[:, None] & ~reset & (d < D)[None, :],
            other=0,
        )
        tl.store(
            S + slot * SD0 + d[None, :] * SD1 + j[:, None] * SD2,
            value + old,
            (j < 3)[:, None] & (d < D)[None, :],
        )


def causal_conv(
    x, weight, state, starts, slots, reset, *, parallel=False, round_before_silu=False
):
    """State is [slots, channels, 3], possibly a strided view of a page."""
    assert x.stride(1) == 1 and weight.is_contiguous()
    assert weight.shape == (x.shape[-1], 4)
    assert state.shape[1:] == (x.shape[-1], 3)
    if parallel:
        y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        _parallel_conv[(triton.cdiv(x.shape[0], 16), triton.cdiv(x.shape[-1], 128))](
            x,
            weight,
            state,
            y,
            starts,
            slots,
            reset,
            x.shape[0],
            x.shape[-1],
            x.stride(0),
            *state.stride(),
            round_before_silu,
            slots.numel(),
            triton.next_power_of_2(slots.numel()),
            16,
            128,
            num_warps=4,
        )
        # Readers finish before in-place state writes, including short chunks.
        _commit_conv[(slots.numel(), triton.cdiv(x.shape[-1], 128))](
            x,
            state,
            starts,
            slots,
            reset,
            x.shape[-1],
            x.stride(0),
            *state.stride(),
            128,
            num_warps=4,
        )
        return y
    y = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    _conv[(slots.numel(), triton.cdiv(x.shape[-1], 128))](
        x,
        weight,
        state,
        y,
        starts,
        slots,
        reset,
        x.shape[-1],
        x.stride(0),
        *state.stride(),
        round_before_silu,
        128,
    )
    return y
