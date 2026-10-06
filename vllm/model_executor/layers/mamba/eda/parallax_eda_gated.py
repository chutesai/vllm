# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode preparation and K-major recurrence in one kernel."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _decode_gated(
    M,
    E,
    F,
    B,
    C,
    A,
    DT,
    S,
    OUT,
    STARTS,
    SLOTS,
    RESET,
    H: tl.constexpr,
    D: tl.constexpr,
    NS: tl.constexpr,
    MS: tl.constexpr,
    ES: tl.constexpr,
    FS: tl.constexpr,
    BS: tl.constexpr,
    CS: tl.constexpr,
    SS: tl.constexpr,
    SH: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    GRID_ORDER: tl.constexpr,
    DUMP: tl.constexpr = False,
    QDEBUG=None,
    KDEBUG=None,
    EDEBUG=None,
    GDEBUG=None,
    BDEBUG=None,
    CDEBUG=None,
):
    sh = tl.program_id(0)
    if GRID_ORDER == "sequence":
        seq, head = sh // H, sh % H
    else:
        seq, head = sh % NS, sh // NS
    kk = tl.arange(0, BK)
    vv = tl.program_id(1) * BV + tl.arange(0, BV)
    begin, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
    slot = tl.load(SLOTS + seq).to(tl.int64)
    live = slot >= 0
    reset = tl.load(RESET + seq)
    ptr = S + slot * SS + head * SH + kk[:, None] * SK + vv[None, :] * SV
    state = tl.load(
        ptr, live & ~reset & (kk < D)[:, None] & (vv < D)[None, :], other=0
    ).to(tl.float32)
    for t in range(begin, end):
        base = t * MS + head * D
        q = tl.load(M + base + kk, kk < D, other=0).to(tl.float32)
        k = tl.load(M + base + H * D + kk, kk < D, other=0).to(tl.float32)
        e = tl.load(E + t * ES + head * D + kk, kk < D, other=0).to(tl.float32)
        q = (
            (q / tl.maximum(tl.sqrt(tl.sum(q * q, 0)), 1e-12))
            .to(M.dtype.element_ty)
            .to(tl.float32)
        )
        k = (
            (k / tl.maximum(tl.sqrt(tl.sum(k * k, 0)), 1e-12))
            .to(M.dtype.element_ty)
            .to(tl.float32)
        )
        e = (
            (e / tl.maximum(tl.sqrt(tl.sum(e * e, 0)), 1e-12))
            .to(E.dtype.element_ty)
            .to(tl.float32)
        )
        u = tl.load(F + t * FS + head * D + kk, kk < D, other=0).to(tl.float32)
        u += tl.load(DT + head * D + kk, kk < D, other=0).to(tl.float32)
        a = tl.exp(tl.load(A + head).to(tl.float32)) / 5.0
        softplus = tl.where(u > 20.0, u, tl.log(1.0 + tl.exp(u)))
        decay = -5.0 + 5.0 * tl.exp(-a * softplus)
        beta = tl.load(B + t * BS + head).to(tl.float32)
        gamma = tl.load(C + t * CS + head).to(tl.float32)
        beta = 1.0 / (1.0 + tl.exp(-beta))
        gamma = 1.0 / (1.0 + tl.exp(-gamma))
        if DUMP:  # noqa: SIM102 -- remove optional debug pointers at compile time
            if tl.program_id(1) == 0:
                off = (t * H + head) * D + kk
                tl.store(QDEBUG + off, q, kk < D)
                tl.store(KDEBUG + off, k, kk < D)
                tl.store(EDEBUG + off, e, kk < D)
                tl.store(GDEBUG + off, decay, kk < D)
                tl.store(BDEBUG + t * H + head, beta)
                tl.store(CDEBUG + t * H + head, gamma)
        v = tl.load(M + base + 2 * H * D + vv, vv < D, other=0).to(tl.float32)
        state = state * tl.exp(decay)[:, None]
        read = tl.sum(e[:, None] * state, 0)
        state = state - (gamma * e)[:, None] * read[None, :]
        residual = v - tl.sum(k[:, None] * state, 0)
        state = state + (beta * k)[:, None] * residual[None, :]
        out = tl.sum((q * SCALE)[:, None] * state, 0)
        tl.store(OUT + (t * H + head) * D + vv, tl.where(live, out, 0), vv < D)
        state = state.to(S.dtype.element_ty).to(tl.float32)
    tl.store(ptr, state, live & (end > begin) & (kk < D)[:, None] & (vv < D)[None, :])


def decode_gated(
    mixed,
    erase,
    f,
    b,
    c,
    a_log,
    dt_bias,
    state,
    starts,
    slots,
    reset,
    *,
    block_v=64,
    num_warps=4,
    grid_order="sequence",
):
    """Use live request metadata and retain BF16 normalization rounding."""
    h, d = state.shape[1:3]
    out = torch.zeros((mixed.shape[0], h, d), device=mixed.device, dtype=mixed.dtype)
    _decode_gated[(slots.numel() * h, triton.cdiv(d, block_v))](
        mixed,
        erase,
        f,
        b,
        c,
        a_log,
        dt_bias,
        state,
        out,
        starts,
        slots,
        reset,
        h,
        d,
        slots.numel(),
        mixed.stride(0),
        erase.stride(0),
        f.stride(0),
        b.stride(0),
        c.stride(0),
        *state.stride(),
        triton.next_power_of_2(d),
        block_v,
        d**-0.5,
        grid_order,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return out
