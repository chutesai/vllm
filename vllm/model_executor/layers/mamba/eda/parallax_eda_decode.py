# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K-major EDA scan and graph-safe single-token decode, with FP32 arithmetic."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _scan(
    Q,
    K,
    V,
    E,
    G,
    B,
    C,
    S,
    OUT,
    STARTS,
    SLOTS,
    RESET,
    H: tl.constexpr,
    D: tl.constexpr,
    SS: tl.constexpr,
    SH: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    ROUND_EACH: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    kk = tl.arange(0, BK)
    vv = tl.program_id(2) * BV + tl.arange(0, BV)
    begin, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
    slot = tl.load(SLOTS + seq).to(tl.int64)
    live = slot >= 0
    reset = tl.load(RESET + seq)
    ptr = S + slot * SS + head * SH + kk[:, None] * SK + vv[None, :] * SV
    state = tl.load(
        ptr, live & ~reset & (kk < D)[:, None] & (vv < D)[None, :], other=0
    ).to(tl.float32)
    for t in range(begin, end):
        base = (t * H + head) * D
        q = tl.load(Q + base + kk, kk < D, other=0).to(tl.float32)
        k = tl.load(K + base + kk, kk < D, other=0).to(tl.float32)
        e = tl.load(E + base + kk, kk < D, other=0).to(tl.float32)
        g = tl.load(G + base + kk, kk < D, other=0).to(tl.float32)
        v = tl.load(V + base + vv, vv < D, other=0).to(tl.float32)
        beta = tl.load(B + t * H + head).to(tl.float32)
        gamma = tl.load(C + t * H + head).to(tl.float32)
        state = state * tl.exp(g)[:, None]
        read = tl.sum(e[:, None] * state, 0)
        state = state - (gamma * e)[:, None] * read[None, :]
        residual = v - tl.sum(k[:, None] * state, 0)
        state = state + (beta * k)[:, None] * residual[None, :]
        out = tl.sum((q * SCALE)[:, None] * state, 0)
        tl.store(OUT + base + vv, tl.where(live, out, 0), vv < D)
        if ROUND_EACH:
            state = state.to(S.dtype.element_ty).to(tl.float32)
    tl.store(ptr, state, live & (end > begin) & (kk < D)[:, None] & (vv < D)[None, :])


def recurrent(
    q,
    k,
    v,
    e,
    g,
    beta,
    gamma,
    state,
    starts,
    slots,
    reset,
    *,
    round_each_step=False,
    block_v=16,
    num_warps=4,
):
    """Packed [T,H,D] inputs; invalid slots emit zeros and preserve the cache."""
    inputs = [x.contiguous() for x in (q, k, v, e, g, beta, gamma)]
    h, d = q.shape[1:]
    output = torch.zeros_like(v)
    _scan[(slots.numel(), h, triton.cdiv(d, block_v))](
        *inputs,
        state,
        output,
        starts,
        slots,
        reset,
        h,
        d,
        *state.stride(),
        triton.next_power_of_2(d),
        block_v,
        d**-0.5,
        round_each_step,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return output


def decode(*args, block_v=16, num_warps=4):
    """Round reduced storage once after each token; no host metadata reads."""
    return recurrent(*args, round_each_step=True, block_v=block_v, num_warps=num_warps)
