# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in FP16 state storage with FP32 updates. This changes state precision.

round_each_step simulates token-by-token decode rounding during fixed-token
prefill quality probes. It is off in the serving prefill path.
"""

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
):
    seq, head = tl.program_id(0), tl.program_id(1)
    kt = tl.arange(0, D)
    vt = tl.program_id(2) * BV + tl.arange(0, BV)
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
        raise ValueError("Fused decode requires FP32 128x128 recurrent heads")
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
    out = torch.empty((conv.shape[0], h, d), device=conv.device, dtype=conv.dtype)
    _decode_gated[(slots.numel(), h, d // block_v)](
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
        num_warps=num_warps,
    )
    return out


@triton.jit
def _scan(
    Q,
    K,
    V,
    G,
    E,
    W,
    S,
    OUT,
    STARTS,
    SLOTS,
    RESET,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    SS0: tl.constexpr,
    ROUND_STATE: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    vt = tl.program_id(2) * BV + tl.arange(0, BV)
    kt = tl.arange(0, BK)
    start = tl.load(STARTS + seq)
    end = tl.load(STARTS + seq + 1)
    slot = tl.load(SLOTS + seq)
    if slot >= 0 and end > start:
        reset = tl.load(RESET + seq)
        addr = slot * SS0 + (head * DK + kt[:, None]) * DV + vt[None, :]
        mask = (kt[:, None] < DK) & (vt[None, :] < DV)
        state = tl.load(S + addr, mask & ~reset, other=0).to(tl.float32)
        for token in range(start, end):
            ki = (token * H + head) * DK + kt
            vi = (token * H + head) * DV + vt
            q = tl.load(Q + ki, kt < DK, other=0).to(tl.float32) * SCALE
            k = tl.load(K + ki, kt < DK, other=0).to(tl.float32)
            decay = tl.exp(tl.load(G + ki, kt < DK, other=0).to(tl.float32))
            erase = tl.load(E + ki, kt < DK, other=0).to(tl.float32)
            v = tl.load(V + vi, vt < DV, other=0).to(tl.float32)
            write = tl.load(W + vi, vt < DV, other=0).to(tl.float32)
            state = state * decay[:, None]
            residual = write * v - tl.sum(state * (erase * k)[:, None], axis=0)
            state = state + k[:, None] * residual[None, :]
            y = tl.sum(state * q[:, None], axis=0)
            tl.store(OUT + vi, y, vt < DV)
            if ROUND_STATE:
                state = state.to(tl.float16).to(tl.float32)
        tl.store(S + addr, state, mask)


def recurrent(
    q,
    k,
    v,
    log_decay,
    erase,
    write,
    state,
    starts,
    slots,
    reset,
    *,
    scale=None,
    out=None,
    block_v=16,
    round_each_step=False,
):
    """Update FP32 state in place, returning [tokens, heads, value_dim].

    starts is int32 cumulative sequence length; slots is int32 request-slot
    indirection. Negative slots are inactive graph-padding rows (output zero).
    reset marks new requests; empty sequences do not modify the state.
    State slots MUST be unique for active sequences; validate at scheduling.
    """
    if q.ndim != 3 or state.ndim != 4:
        raise ValueError("expected token-major Q and slot-major 4D state")
    t, h, dk = q.shape
    dv = v.shape[-1]
    if state.dtype not in (torch.float32, torch.float16) or state.shape[1:] != (
        h,
        dk,
        dv,
    ):
        raise ValueError("FP32 [slots, heads, key_dim, value_dim] state required")
    if state.stride()[1:] != (dk * dv, dv, 1):
        raise ValueError("state pages must be contiguous within each slot")
    if not state.is_cuda or state.device != q.device:
        raise ValueError("state must be on the input device")
    for x in (q, k, v, log_decay, erase, write, starts, slots, reset):
        if not x.is_cuda or not x.is_contiguous() or x.device != q.device:
            raise ValueError("inputs must be contiguous and on the same CUDA device")
    if k.shape != q.shape or log_decay.shape != q.shape or erase.shape != q.shape:
        raise ValueError("key-axis tensor shape mismatch")
    if v.shape != (t, h, dv) or write.shape != v.shape:
        raise ValueError("value-axis tensor shape mismatch")
    if starts.numel() != slots.numel() + 1 or reset.shape != slots.shape:
        raise ValueError("sequence metadata shape mismatch")
    if (
        starts.dtype != torch.int32
        or slots.dtype != torch.int32
        or reset.dtype != torch.bool
    ):
        raise ValueError("metadata must be int32/int32/bool")
    if out is None:
        out = torch.zeros_like(v)
    else:
        if (
            out.shape != v.shape
            or out.dtype != v.dtype
            or out.device != v.device
            or not out.is_contiguous()
        ):
            raise ValueError("output layout mismatch")
        out.zero_()
    if block_v not in (4, 8, 16, 32):
        raise ValueError("Unsupported recurrent value tile")
    _scan[(slots.numel(), h, triton.cdiv(dv, block_v))](
        q,
        k,
        v,
        log_decay,
        erase,
        write,
        state,
        out,
        starts,
        slots,
        reset,
        h,
        dk,
        dv,
        triton.next_power_of_2(dk),
        block_v,
        dk**-0.5 if scale is None else scale,
        state.stride(0),
        round_each_step,
        num_warps=4,
    )
    return out


def reference(
    q, k, v, log_decay, erase, write, state, starts, slots, reset, *, scale=None
):
    """Independent PyTorch recurrence, including ragged and reset behavior."""
    state = state.clone()
    out = torch.zeros_like(v)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    for seq, slot in enumerate(slots.cpu().tolist()):
        a, b = starts[seq : seq + 2].cpu().tolist()
        if slot < 0 or a == b:
            continue
        s = torch.zeros_like(state[slot]) if reset[seq] else state[slot].clone()
        for t in range(a, b):
            s = log_decay[t].float().exp()[..., None] * s
            read = torch.einsum("hk,hkv->hv", erase[t].float() * k[t].float(), s)
            delta = write[t].float() * v[t].float() - read
            s = s + k[t].float()[..., None] * delta[:, None, :]
            out[t] = torch.einsum("hk,hkv->hv", q[t].float() * scale, s)
        state[slot] = s
    return out, state
