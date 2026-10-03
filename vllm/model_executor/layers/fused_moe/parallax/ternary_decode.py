# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-row-count MoE fallback: two-bit weights, fused ReLU², no FP4 claim."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _matvec(
    X,
    W,
    A,
    IDS,
    GATES,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    TOP: tl.constexpr,
    FIRST: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PACKED_VECTOR: tl.constexpr = False,
):
    route = tl.program_id(0)
    token = route // TOP
    expert = tl.load(IDS + route)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    xoff = token * K if FIRST else route * K
    if PACKED_VECTOR:
        ks = tl.arange(0, BK // 4)
        packed = tl.load(
            W + (expert * N + ns[:, None]) * (K // 4) + ks[None, :],
            (ns[:, None] < N) & (ks[None, :] < K // 4),
            other=0,
        )
        dot = tl.full((BN, BK // 4), 0, tl.float32)
        for shift in tl.static_range(4):
            x = tl.load(X + xoff + ks * 4 + shift, ks < K // 4, other=0).to(tl.float32)
            code = (packed >> (shift * 2)) & 3
            w = tl.where(code == 1, 1.0, tl.where(code == 2, -1.0, 0.0))
            dot += w * x[None, :]
        val = tl.sum(dot, axis=1) * tl.load(A + expert)
    else:
        ks = tl.arange(0, BK)
        x = tl.load(X + xoff + ks, ks < K, other=0).to(tl.float32)
        packed = tl.load(
            W + (expert * N + ns[:, None]) * (K // 4) + ks[None, :] // 4,
            (ns[:, None] < N) & (ks[None, :] < K),
            other=0,
        )
        code = (packed >> ((ks[None, :] % 4) * 2)) & 3
        w = tl.where(code == 1, 1.0, tl.where(code == 2, -1.0, 0.0))
        val = tl.sum(w * x[None, :], axis=1) * tl.load(A + expert)
    if FIRST:
        val = tl.maximum(val, 0.0)
        val = val * val
    else:
        val = val * tl.load(GATES + route)
    tl.store(Y + route * N + ns, val, ns < N)


@triton.jit
def _combine(
    X, Y, N: tl.constexpr, TOP: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr
):
    token = tl.program_id(0)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    ts = tl.arange(0, BT)
    x = tl.load(
        X + (token * TOP + ts[:, None]) * N + ns[None, :],
        (ts[:, None] < TOP) & (ns[None, :] < N),
        other=0,
    ).to(tl.float32)
    tl.store(Y + token * N + ns, tl.sum(x, axis=0), ns < N)


def pack(codes):
    if codes.shape[-1] % 4 or codes.dtype != torch.int8:
        raise ValueError("int8 ternary codes with K divisible by four required")
    if not bool(((codes == -1) | (codes == 0) | (codes == 1)).all()):
        raise ValueError("non-ternary weight")
    q = torch.where(codes < 0, 2, codes).to(torch.uint8)
    return (
        q[..., 0::4] | q[..., 1::4] << 2 | q[..., 2::4] << 4 | q[..., 3::4] << 6
    ).contiguous()


def moe(x, up, down, alpha_up, alpha_down, ids, gates, *, vector_tiles=None):
    """Expert-only latent-in/latent-out path; assumes validated router indices."""
    tokens, latent = x.shape
    experts, hidden, _ = up.shape
    top = ids.shape[1]
    if (
        down.shape != (experts, latent, hidden // 4)
        or up.shape[-1] * 4 != latent
        or ids.shape != gates.shape
        or ids.shape[0] != tokens
    ):
        raise ValueError("expert or routing shape mismatch")
    if up.dtype != torch.uint8 or down.dtype != torch.uint8:
        raise ValueError("two-bit packed uint8 weights required")
    for a in (x, up, down, alpha_up, alpha_down, ids, gates):
        if not a.is_cuda or not a.is_contiguous() or a.device != x.device:
            raise ValueError("contiguous inputs on one CUDA device required")
    h = torch.empty(tokens * top, hidden, device=x.device, dtype=x.dtype)
    partial = torch.empty(tokens * top, latent, device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    bu, bd = vector_tiles or (4, 4)
    _matvec[(tokens * top, triton.cdiv(hidden, bu))](
        x,
        up,
        alpha_up,
        ids,
        gates,
        h,
        hidden,
        latent,
        top,
        True,
        bu,
        triton.next_power_of_2(latent),
        vector_tiles is not None,
        num_warps=4,
    )
    _matvec[(tokens * top, triton.cdiv(latent, bd))](
        h,
        down,
        alpha_down,
        ids,
        gates,
        partial,
        latent,
        hidden,
        top,
        False,
        bd,
        triton.next_power_of_2(hidden),
        vector_tiles is not None,
        num_warps=4,
    )
    _combine[(tokens, triton.cdiv(latent, 128))](
        partial, y, latent, top, 128, triton.next_power_of_2(top)
    )
    return y
