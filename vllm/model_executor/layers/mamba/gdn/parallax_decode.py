# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental GDN scheduling; unchanged state arithmetic and precision."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _decode_gated(
    CONV,
    BW,
    F,
    A,
    DT,
    S,
    OUT,
    STARTS,
    SLOTS,
    RESET,
    H: tl.constexpr,
    D: tl.constexpr,
    SS: tl.constexpr,
    BWS: tl.constexpr,
    FS: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NS: tl.constexpr,
    LOAD_CACHE: tl.constexpr,
    STORE_CACHE: tl.constexpr,
    ORDER: tl.constexpr,
):
    tiles = D // BV
    pid = tl.program_id(0)
    if ORDER == 0:
        seq = pid % NS
        head = (pid // NS) % H
        tile = pid // (NS * H)
    elif ORDER == 1:
        tile = pid % tiles
        head = (pid // tiles) % H
        seq = pid // (tiles * H)
    elif ORDER == 2:
        tile = pid % tiles
        seq = (pid // tiles) % NS
        head = pid // (tiles * NS)
    else:
        head = pid % H
        tile = (pid // H) % tiles
        seq = pid // (H * tiles)
    kt = tl.arange(0, D)
    vt = tile * BV + tl.arange(0, BV)
    token = tl.load(STARTS + seq)
    end = tl.load(STARTS + seq + 1)
    slot = tl.load(SLOTS + seq)
    # Graphs capture a larger token buffer than some replays use. Zero only
    # rows beyond the packed active length, avoiding races with interior gaps.
    if seq >= tl.load(STARTS + NS) and seq < NT:
        tl.store(OUT + (seq * H + head) * D + vt, 0.0)
    if end > token and slot >= 0:
        reset = tl.load(RESET + seq)
        width = H * D
        off = token * width * 3 + head * D
        q = tl.load(CONV + off + kt).to(tl.float32)
        k = tl.load(CONV + off + width + kt).to(tl.float32)
        v = tl.load(CONV + off + 2 * width + vt).to(tl.float32)
        q = (
            (q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6))
            .to(CONV.dtype.element_ty)
            .to(tl.float32)
        )
        k = (
            (k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6))
            .to(CONV.dtype.element_ty)
            .to(tl.float32)
        )
        erase = (
            tl.sigmoid(tl.load(BW + token * BWS + head * D + kt).to(tl.float32))
            .to(CONV.dtype.element_ty)
            .to(tl.float32)
        )
        write = (
            tl.sigmoid(tl.load(BW + token * BWS + width + head * D + vt).to(tl.float32))
            .to(CONV.dtype.element_ty)
            .to(tl.float32)
        )
        z = tl.load(F + token * FS + head * D + kt).to(tl.float32) + tl.load(
            DT + head * D + kt
        ).to(tl.float32)
        softplus = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(z)))
        decay = tl.exp(-tl.exp(tl.load(A + head)) * softplus)
        addr = slot * SS + (head * D + kt[:, None]) * D + vt[None, :]
        state = (
            tl.load(S + addr, ~reset, other=0, cache_modifier=LOAD_CACHE)
            * decay[:, None]
        )
        residual = write * v - tl.sum(state * (erase * k)[:, None], 0)
        state = state + k[:, None] * residual[None, :]
        y = tl.sum(state * (q * D**-0.5)[:, None], 0)
        tl.store(S + addr, state, cache_modifier=STORE_CACHE)
        tl.store(OUT + (token * H + head) * D + vt, y)
    elif end > token:
        tl.store(OUT + (token * H + head) * D + vt, 0.0)


def decode_gated(
    conv,
    bw,
    f,
    a,
    dt,
    state,
    starts,
    slots,
    reset,
    *,
    block_v=16,
    num_warps=4,
    state_load_cache="",
    state_store_cache="",
    grid_order=0,
):
    """One token per nonempty sequence; fuse gates into the FP32 recurrence.

    The scheduler guarantees decode-only lengths and unique live slots. Unused
    graph-padding output rows and negative live slots are written as zero.
    """
    h, d = state.shape[1:3]
    if (
        d != 128
        or state.shape[3] != d
        or state.dtype not in (torch.float32, torch.float16)
    ):
        raise ValueError("Fused decode requires FP32 or FP16 128x128 recurrent heads")
    if state.stride()[1:] != (d * d, d, 1) or not conv.is_contiguous():
        raise ValueError("Contiguous inputs and per-slot recurrent pages required")
    if conv.shape[0] > slots.numel():
        raise ValueError("Decode token capacity cannot exceed slot capacity")
    if block_v not in (4, 8, 16, 32, 64, 128) or num_warps not in (1, 2, 4, 8):
        raise ValueError("Unsupported recurrent value tile/warp count")
    if state_load_cache not in ("", ".ca", ".cg") or state_store_cache not in (
        "",
        ".wb",
        ".cs",
        ".wt",
    ):
        raise ValueError("Unsupported state cache policy")
    if grid_order not in (0, 1, 2, 3):
        raise ValueError("Unsupported GDN grid order")
    out = torch.empty((conv.shape[0], h, d), device=conv.device, dtype=conv.dtype)
    _decode_gated[(slots.numel() * h * (d // block_v),)](
        conv,
        bw,
        f,
        a,
        dt,
        state,
        out,
        starts,
        slots,
        reset,
        h,
        d,
        state.stride(0),
        bw.stride(0),
        f.stride(0),
        block_v,
        conv.shape[0],
        slots.numel(),
        state_load_cache,
        state_store_cache,
        grid_order,
        num_warps=num_warps,
    )
    return out
