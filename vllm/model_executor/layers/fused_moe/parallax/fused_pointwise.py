# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving pointwise fusion, preserving the reference BF16 rounding points."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _route(
    LOGITS,
    BETA,
    IDS,
    GATES,
    E: tl.constexpr,
    TOP: tl.constexpr,
    SCALE: tl.constexpr,
    BE: tl.constexpr,
    BT: tl.constexpr,
    PARTS: tl.constexpr = 1,
    TOKENS: tl.int32 = 1,
    SORT: tl.constexpr = False,
):
    token = tl.program_id(0)
    es = tl.arange(0, BE)
    rs = tl.arange(0, BT)
    if PARTS == 1:
        logits = tl.load(LOGITS + token * E + es, es < E, other=-float("inf"))
    else:
        ps = tl.arange(0, triton.next_power_of_2(PARTS))
        values = tl.load(
            LOGITS + (ps[:, None] * TOKENS + token) * E + es[None, :],
            (ps[:, None] < PARTS) & (es[None, :] < E),
            other=0.0,
        )
        logits = tl.sum(values, axis=0)
        logits = tl.where(es < E, logits, -float("inf"))
    scores = logits - tl.load(BETA + es, es < E, other=0.0)
    if SORT:
        # Preserve every FP32 score bit and the existing lower-ID tie break.
        bits = tl.where(scores == 0.0, 0.0, scores).to(tl.uint32, bitcast=True)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
        keys = (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - es.to(tl.uint32)).to(
            tl.uint64
        )
        best = tl.topk(keys, BT)
        indices = (0xFFFFFFFF - (best & 0xFFFFFFFF)).to(tl.int32)
        selected = tl.gather(logits, indices, 0)
        gates = tl.where(rs < TOP, tl.sigmoid(selected), 0.0)
    else:
        indices = tl.full((BT,), 0, tl.int32)
        gates = tl.full((BT,), 0, tl.float32)
        for r in range(TOP):
            peak = tl.max(scores, axis=0)
            chosen = tl.min(
                tl.where((scores == peak) & (es < E), es, 2147483647), axis=0
            )
            selected = tl.sum(tl.where(es == chosen, logits, 0.0), axis=0)
            indices = tl.where(rs == r, chosen, indices)
            gates = tl.where(rs == r, tl.sigmoid(selected), gates)
            scores = tl.where(es == chosen, -float("inf"), scores)
    gates = gates / tl.maximum(tl.sum(gates, axis=0), 1.0e-6) * SCALE
    tl.store(IDS + token * TOP + rs, indices, rs < TOP)
    tl.store(GATES + token * TOP + rs, gates, rs < TOP)


def route(logits, beta, top, scale=1.0, *, fast_sort=False):
    if (
        logits.dtype != torch.float32
        or beta.dtype != torch.float32
        or logits.ndim != 2
        or beta.shape != (logits.shape[1],)
        or top < 1
        or top > logits.shape[1]
    ):
        raise ValueError("FP32 routing logits and balancing offsets required")
    ids = torch.empty((logits.shape[0], top), device=logits.device, dtype=torch.int32)
    gates = torch.empty_like(ids, dtype=torch.float32)
    _route[(logits.shape[0],)](
        logits,
        beta,
        ids,
        gates,
        logits.shape[1],
        top,
        scale,
        triton.next_power_of_2(logits.shape[1]),
        triton.next_power_of_2(top),
        SORT=fast_sort,
        num_warps=4,
    )
    return ids, gates


@triton.jit
def _bar(
    SOURCES,
    GAMMA,
    QUERY,
    OUT,
    POST_GAMMA,
    POST_NORM: tl.constexpr,
    D: tl.constexpr,
    NS: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
):
    token = tl.program_id(0)
    d = tl.arange(0, BD)
    s = tl.arange(0, BN)
    gamma = tl.load(GAMMA + d, d < D, other=0).to(tl.float32)
    query = tl.load(QUERY + d, d < D, other=0).to(tl.float32)
    logits = tl.full((BN,), -float("inf"), tl.float32)
    for i in tl.static_range(NS):
        x = tl.load(SOURCES[i] + token * D + d, d < D, other=0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=0) / D + 1e-6)
        z = (x * inv * gamma).to(OUT.dtype.element_ty).to(tl.float32)
        score = tl.sum(z * query, axis=0).to(OUT.dtype.element_ty).to(tl.float32)
        logits = tl.where(s == i, score, logits)
    probs = tl.exp(logits - tl.max(logits, axis=0))
    probs = probs / tl.sum(probs, axis=0)
    acc = tl.full((BD,), 0, tl.float32)
    for i in tl.static_range(NS):
        weight = tl.sum(tl.where(s == i, probs, 0.0), axis=0).to(OUT.dtype.element_ty)
        x = tl.load(SOURCES[i] + token * D + d, d < D, other=0)
        acc += (x * weight).to(OUT.dtype.element_ty).to(tl.float32)
    if POST_NORM:
        acc = acc.to(OUT.dtype.element_ty).to(tl.float32)
        inv = tl.rsqrt(tl.sum(acc * acc, axis=0) / D + 1e-6)
        gamma = tl.load(POST_GAMMA + d, d < D, other=0).to(tl.float32)
        acc = acc * inv * gamma
    tl.store(OUT + token * D + d, acc, d < D)


@triton.jit
def _bar_scores(
    SOURCES, GAMMA, QUERY, SCORES, D: tl.constexpr, NS: tl.constexpr, BD: tl.constexpr
):
    token, source = tl.program_id(0), tl.program_id(1)
    ptr = SOURCES[0]
    for i in tl.static_range(1, NS):
        ptr = tl.where(source == i, SOURCES[i], ptr)
    d = tl.arange(0, BD)
    x = tl.load(ptr + token * D + d, d < D, other=0).to(tl.float32)
    gamma = tl.load(GAMMA + d, d < D, other=0).to(tl.float32)
    query = tl.load(QUERY + d, d < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + 1e-6)
    z = (x * inv * gamma).to(ptr.dtype.element_ty).to(tl.float32)
    score = tl.sum(z * query, 0).to(ptr.dtype.element_ty).to(tl.float32)
    tl.store(SCORES + token * NS + source, score)


@triton.jit
def _bar_combine(
    SOURCES,
    SCORES,
    OUT,
    POST_GAMMA,
    POST_NORM: tl.constexpr,
    D: tl.constexpr,
    NS: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
):
    token = tl.program_id(0)
    d = tl.arange(0, BD)
    s = tl.arange(0, BN)
    logits = tl.load(SCORES + token * NS + s, s < NS, other=-float("inf"))
    probs = tl.exp(logits - tl.max(logits, 0))
    probs = probs / tl.sum(probs, 0)
    acc = tl.full((BD,), 0, tl.float32)
    for i in tl.static_range(NS):
        weight = tl.sum(tl.where(s == i, probs, 0.0), 0).to(OUT.dtype.element_ty)
        x = tl.load(SOURCES[i] + token * D + d, d < D, other=0)
        acc += (x * weight).to(OUT.dtype.element_ty).to(tl.float32)
    if POST_NORM:
        acc = acc.to(OUT.dtype.element_ty).to(tl.float32)
        inv = tl.rsqrt(tl.sum(acc * acc, 0) / D + 1e-6)
        gamma = tl.load(POST_GAMMA + d, d < D, other=0).to(tl.float32)
        acc = acc * inv * gamma
    tl.store(OUT + token * D + d, acc, d < D)


def bar(sources, gamma, query, post_gamma=None, *, split=False):
    width = sources[0].shape[-1]
    out = torch.empty_like(sources[0])
    if split:
        scores = torch.empty(
            (out.shape[0], len(sources)), device=out.device, dtype=torch.float32
        )
        _bar_scores[(out.shape[0], len(sources))](
            tuple(sources),
            gamma,
            query,
            scores,
            width,
            len(sources),
            triton.next_power_of_2(width),
            num_warps=4,
        )
        _bar_combine[(out.shape[0],)](
            tuple(sources),
            scores,
            out,
            post_gamma,
            post_gamma is not None,
            width,
            len(sources),
            triton.next_power_of_2(width),
            triton.next_power_of_2(len(sources)),
            num_warps=4,
        )
        return out
    _bar[(out.shape[0],)](
        tuple(sources),
        gamma,
        query,
        out,
        post_gamma,
        post_gamma is not None,
        width,
        len(sources),
        triton.next_power_of_2(width),
        triton.next_power_of_2(len(sources)),
        num_warps=4,
    )
    return out


@triton.jit
def _gates(
    CONV,
    BW,
    F,
    A,
    DT,
    Q,
    K,
    V,
    DECAY,
    ERASE,
    WRITE,
    H: tl.constexpr,
    D: tl.constexpr,
    BWS: tl.constexpr,
    FS: tl.constexpr,
    BD: tl.constexpr,
):
    t, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    width = H * D
    q = tl.load(CONV + t * width * 3 + h * D + d, d < D, other=0).to(tl.float32)
    k = tl.load(CONV + t * width * 3 + width + h * D + d, d < D, other=0).to(tl.float32)
    v = tl.load(CONV + t * width * 3 + 2 * width + h * D + d, d < D, other=0)
    q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6)
    k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
    b = tl.load(BW + t * BWS + h * D + d, d < D, other=0).to(tl.float32)
    w = tl.load(BW + t * BWS + width + h * D + d, d < D, other=0).to(tl.float32)
    f = tl.load(F + t * FS + h * D + d, d < D, other=0).to(tl.float32)
    dt = tl.load(DT + h * D + d, d < D, other=0).to(tl.float32)
    a = tl.load(A + h).to(tl.float32)
    z = f + dt
    softplus = tl.maximum(z, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(z)))
    decay = -tl.exp(a) * softplus
    off = t * width + h * D + d
    tl.store(Q + off, q, d < D)
    tl.store(K + off, k, d < D)
    tl.store(V + off, v, d < D)
    tl.store(DECAY + off, decay, d < D)
    tl.store(ERASE + off, tl.sigmoid(b), d < D)
    tl.store(WRITE + off, tl.sigmoid(w), d < D)


def gates(conv, bw, f, a, dt, heads, dim):
    n = conv.shape[0]
    q, k, v, erase, write = [
        torch.empty(n, heads, dim, device=conv.device, dtype=conv.dtype)
        for _ in range(5)
    ]
    decay = torch.empty_like(q, dtype=torch.float32)
    _gates[(n, heads)](
        conv,
        bw,
        f,
        a,
        dt,
        q,
        k,
        v,
        decay,
        erase,
        write,
        heads,
        dim,
        bw.stride(0),
        f.stride(0),
        triton.next_power_of_2(dim),
        num_warps=4,
    )
    return q, k, v, decay, erase, write


@triton.jit
def _norm_gate(Y, G, GAMMA, OUT, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    y = tl.load(Y + row * D + d, d < D, other=0).to(tl.float32)
    g = tl.load(G + row * D + d, d < D, other=0).to(tl.float32)
    gamma = tl.load(GAMMA + d, d < D, other=0).to(tl.float32)
    norm = y * tl.rsqrt(tl.sum(y * y, axis=0) / D + 1e-6) * gamma
    norm = norm.to(OUT.dtype.element_ty).to(tl.float32)
    gate = (g * tl.sigmoid(g)).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + row * D + d, norm * gate, d < D)


def norm_gate(y, g, gamma):
    out = torch.empty_like(y)
    _norm_gate[(y.numel() // y.shape[-1],)](
        y, g, gamma, out, y.shape[-1], triton.next_power_of_2(y.shape[-1]), num_warps=4
    )
    return out
