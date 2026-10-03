# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parallel chunk preparation and output, with an ordered FP32 GDN2 state pass.

For cumulative channel decay G, U=k/G, B=erase*k*G and L=tril(B U^T,-1),
W=(I+L)^-1 B and Z=(I+L)^-1(write*v). Each chunk advances the carried
state by G_last*(S+U^T*(Z-W*S)). Outputs use an independently prepared
linear map of the incoming state. No scalar-gate approximation is made.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _location(CI, STARTS, NS: tl.int32, BS: tl.constexpr, C: tl.constexpr):
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < NS, other=2147483000)
    next_base = ends // C + ss + 1
    seq = tl.sum(((next_base <= CI) & (ss < NS)).to(tl.int32), 0)
    begin = tl.load(STARTS + seq, seq < NS, other=0)
    end = tl.load(STARTS + seq + 1, seq < NS, other=0)
    token = begin + (CI - (begin // C + seq)) * C
    return seq, token, end


@triton.jit
def _prepare(
    Q,
    K,
    V,
    G,
    E,
    WRITE,
    STARTS,
    SLOTS,
    U,
    W,
    Z,
    YS,
    YV,
    LAST,
    BAD,
    H: tl.constexpr,
    D: tl.constexpr,
    NS: tl.int32,
    BS: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
):
    ci, head = tl.program_id(0), tl.program_id(1)
    seq, start, end = _location(ci, STARTS, NS, BS, C)
    slot = tl.load(SLOTS + seq, seq < NS, other=-1)
    if seq < NS and start < end and slot >= 0:
        ts, ds = tl.arange(0, C), tl.arange(0, D)
        tok = start + ts
        off = (tok[:, None] * H + head) * D + ds[None, :]
        valid = tok[:, None] < end
        q = tl.load(Q + off, valid, other=0).to(tl.float32)
        k = tl.load(K + off, valid, other=0).to(tl.float32)
        v = tl.load(V + off, valid, other=0).to(tl.float32)
        erase = tl.load(E + off, valid, other=0).to(tl.float32)
        write = tl.load(WRITE + off, valid, other=0).to(tl.float32)
        cumulative = tl.cumsum(tl.load(G + off, valid, other=0).to(tl.float32), 0)
        bad = (tl.min(cumulative) < -30.0) | (tl.max(cumulative) > 30.0)
        tl.store(BAD + ci * H + head, bad)
        if not bad:
            decay = tl.exp(cumulative)
            u = k * tl.exp(-cumulative)
            b = erase * k * decay
            qg = q * decay * SCALE
            lower = tl.dot(b, tl.trans(u), input_precision="tf32x3")
            lower = tl.where(ts[:, None] > ts[None, :], lower, 0.0)
            identity = (ts[:, None] == ts[None, :]).to(tl.float32)
            inv = identity - lower
            power = tl.dot(lower, lower, input_precision="tf32x3")
            for _ in tl.static_range(1, 4 if C == 16 else 5):
                inv += tl.dot(inv, power, input_precision="tf32x3")
                power = tl.dot(power, power, input_precision="tf32x3")
            w = tl.dot(inv, b, input_precision="tf32x3")
            z = tl.dot(inv, write * v, input_precision="tf32x3")
            att = tl.dot(qg, tl.trans(u), input_precision="tf32x3")
            att = tl.where(ts[:, None] >= ts[None, :], att, 0.0)
            ys = qg - tl.dot(att, w, input_precision="tf32x3")
            yv = tl.dot(att, z, input_precision="tf32x3")
            dst = ((ci * H + head) * C + ts[:, None]) * D + ds[None, :]
            tl.store(U + dst, u)
            tl.store(W + dst, w)
            tl.store(Z + dst, z)
            tl.store(YS + dst, ys)
            tl.store(YV + dst, yv)
            last = tl.sum(tl.where(ts[:, None] == C - 1, decay, 0.0), 0)
            tl.store(LAST + (ci * H + head) * D + ds, last)


@triton.jit
def _states(
    Q,
    K,
    V,
    G,
    E,
    WRITE,
    S,
    OUT,
    STARTS,
    SLOTS,
    RESET,
    U,
    W,
    Z,
    LAST,
    BAD,
    INCOMING,
    H: tl.constexpr,
    D: tl.constexpr,
    SS: tl.constexpr,
    C: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
):
    seq, head = tl.program_id(0), tl.program_id(1)
    begin, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
    slot = tl.load(SLOTS + seq)
    if slot >= 0 and end > begin:
        ks = tl.arange(0, D)
        vs = tl.program_id(2) * BV + tl.arange(0, BV)
        ts = tl.arange(0, C)
        addr = slot * SS + (head * D + ks[:, None]) * D + vs[None, :]
        state = tl.load(S + addr, ~tl.load(RESET + seq), other=0)
        for j in range(tl.cdiv(end - begin, C)):
            ci = begin // C + seq + j
            saved = ((ci * H + head) * D + ks[:, None]) * D + vs[None, :]
            if tl.load(BAD + ci * H + head):
                for token in range(begin + j * C, tl.minimum(begin + (j + 1) * C, end)):
                    ki = (token * H + head) * D + ks
                    vi = (token * H + head) * D + vs
                    q = tl.load(Q + ki).to(tl.float32) * SCALE
                    k = tl.load(K + ki).to(tl.float32)
                    decay = tl.exp(tl.load(G + ki).to(tl.float32))
                    erase = tl.load(E + ki).to(tl.float32)
                    v = tl.load(V + vi).to(tl.float32)
                    write = tl.load(WRITE + vi).to(tl.float32)
                    state *= decay[:, None]
                    residual = write * v - tl.sum(state * (erase * k)[:, None], 0)
                    state += k[:, None] * residual[None, :]
                    tl.store(OUT + vi, tl.sum(state * q[:, None], 0))
            else:
                tl.store(INCOMING + saved, state)
                ko = ((ci * H + head) * C + ts[:, None]) * D + ks[None, :]
                vo = ((ci * H + head) * C + ts[:, None]) * D + vs[None, :]
                u = tl.load(U + ko)
                w = tl.load(W + ko)
                z = tl.load(Z + vo)
                residual = z - tl.dot(w, state, input_precision="tf32x3")
                state += tl.dot(tl.trans(u), residual, input_precision="tf32x3")
                state *= tl.load(LAST + (ci * H + head) * D + ks)[:, None]
        tl.store(S + addr, state)


@triton.jit
def _outputs(
    YS,
    YV,
    INCOMING,
    BAD,
    OUT,
    STARTS,
    SLOTS,
    H: tl.constexpr,
    D: tl.constexpr,
    NS: tl.int32,
    BS: tl.constexpr,
    C: tl.constexpr,
    BV: tl.constexpr,
):
    ci, head = tl.program_id(0), tl.program_id(1)
    seq, start, end = _location(ci, STARTS, NS, BS, C)
    slot = tl.load(SLOTS + seq, seq < NS, other=-1)
    if seq < NS and start < end and slot >= 0:  # noqa: SIM102 -- guard invalid chunk reads
        if not tl.load(BAD + ci * H + head):
            ts, ks = tl.arange(0, C), tl.arange(0, D)
            vs = tl.program_id(2) * BV + tl.arange(0, BV)
            coef = tl.load(YS + ((ci * H + head) * C + ts[:, None]) * D + ks[None, :])
            state = tl.load(
                INCOMING + ((ci * H + head) * D + ks[:, None]) * D + vs[None, :]
            )
            value = tl.load(YV + ((ci * H + head) * C + ts[:, None]) * D + vs[None, :])
            y = tl.dot(coef, state, input_precision="tf32x3") + value
            tl.store(
                OUT + ((start + ts[:, None]) * H + head) * D + vs[None, :],
                y,
                start + ts[:, None] < end,
            )


def parallel_prefill(
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
    chunk=32,
    state_tile=32,
):
    """Packed ragged sequences, unique live slots, no host reads of GPU metadata."""
    if q.shape[-1] != 128 or v.shape != q.shape or state.dtype != torch.float32:
        raise ValueError("Parallel GDN2 requires K=V=128 and FP32 state")
    if state.stride()[1:] != (128 * 128, 128, 1):
        raise ValueError("Contiguous state pages required")
    if any(
        not x.is_contiguous()
        for x in (q, k, v, log_decay, erase, write, starts, slots, reset)
    ):
        raise ValueError("Contiguous packed inputs required")
    t, h, d = q.shape
    if chunk not in (16, 32) or state_tile not in (16, 32, 64):
        raise ValueError("Unsupported chunk or state tile")
    c = chunk
    cap = triton.cdiv(t, c) + slots.numel()
    options = {"device": q.device, "dtype": torch.float32}
    u, w, z, ys, yv = [torch.empty((cap, h, c, d), **options) for _ in range(5)]
    last = torch.empty((cap, h, d), **options)
    bad = torch.empty((cap, h), device=q.device, dtype=torch.bool)
    incoming = torch.empty((cap, h, d, d), **options)
    out = torch.zeros_like(v)
    _prepare[(cap, h)](
        q,
        k,
        v,
        log_decay,
        erase,
        write,
        starts,
        slots,
        u,
        w,
        z,
        ys,
        yv,
        last,
        bad,
        h,
        d,
        slots.numel(),
        triton.next_power_of_2(slots.numel()),
        c,
        d**-0.5,
        num_warps=4,
        num_stages=1,
    )
    _states[(slots.numel(), h, d // state_tile)](
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
        u,
        w,
        z,
        last,
        bad,
        incoming,
        h,
        d,
        state.stride(0),
        c,
        state_tile,
        d**-0.5,
        num_warps=4,
        num_stages=1,
    )
    _outputs[(cap, h, 4)](
        ys,
        yv,
        incoming,
        bad,
        out,
        starts,
        slots,
        h,
        d,
        slots.numel(),
        triton.next_power_of_2(slots.numel()),
        c,
        32,
        num_warps=4,
        num_stages=1,
    )
    return out
