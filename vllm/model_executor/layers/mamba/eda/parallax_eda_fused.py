# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP32 EDA preparation and output gates with explicit BF16 rounding."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _prepare(
    M,
    E,
    F,
    B,
    C,
    A,
    DT,
    QO,
    KO,
    VO,
    EO,
    GO,
    BO,
    CO,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    CS: tl.constexpr,
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    col = tl.arange(0, BD)
    token, head = row // H, row % H
    offset = row[:, None] * D + col[None, :]
    mask = (row < T * H)[:, None] & (col < D)[None, :]
    mixed = token[:, None] * 3 * H * D + head[:, None] * D + col[None, :]
    q = tl.load(M + mixed, mask, other=0).to(tl.float32)
    k = tl.load(M + mixed + H * D, mask, other=0).to(tl.float32)
    v = tl.load(M + mixed + 2 * H * D, mask, other=0)
    e = tl.load(E + offset, mask, other=0).to(tl.float32)
    q = q / tl.maximum(tl.sqrt(tl.sum(q * q, 1)), 1e-12)[:, None]
    k = k / tl.maximum(tl.sqrt(tl.sum(k * k, 1)), 1e-12)[:, None]
    e = e / tl.maximum(tl.sqrt(tl.sum(e * e, 1)), 1e-12)[:, None]
    u = tl.load(F + offset, mask, other=0).to(tl.float32)
    u += tl.load(DT + head[:, None] * D + col[None, :], col[None, :] < D, other=0).to(
        tl.float32
    )
    a = tl.exp(tl.load(A + head).to(tl.float32)) / 5.0
    softplus = tl.where(u > 20.0, u, tl.log(1.0 + tl.exp(u)))
    decay = -5.0 + 5.0 * tl.exp(-a[:, None] * softplus)
    beta = tl.load(B + token * BS + head, row < T * H, other=0).to(tl.float32)
    gamma = tl.load(C + token * CS + head, row < T * H, other=0).to(tl.float32)
    tl.store(QO + offset, q, mask)
    tl.store(KO + offset, k, mask)
    tl.store(VO + offset, v, mask)
    tl.store(EO + offset, e, mask)
    tl.store(GO + offset, decay, mask)
    tl.store(BO + row, 1.0 / (1.0 + tl.exp(-beta)), row < T * H)
    tl.store(CO + row, 1.0 / (1.0 + tl.exp(-gamma)), row < T * H)


def gates(mixed, erase, f, b, c, a_log, dt_bias, h, d):
    t = mixed.shape[0]
    q, k, v, e = [
        torch.empty((t, h, d), device=mixed.device, dtype=mixed.dtype) for _ in range(4)
    ]
    g = torch.empty((t, h, d), device=mixed.device, dtype=torch.float32)
    beta, gamma = [
        torch.empty((t, h), device=mixed.device, dtype=torch.float32) for _ in range(2)
    ]
    _prepare[(triton.cdiv(t * h, 4),)](
        mixed,
        erase,
        f,
        b,
        c,
        a_log,
        dt_bias,
        q,
        k,
        v,
        e,
        g,
        beta,
        gamma,
        t,
        h,
        d,
        triton.next_power_of_2(d),
        4,
        b.stride(0),
        c.stride(0),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q, k, v, e, g, beta, gamma


@triton.jit
def _output(
    Y,
    GATE,
    W,
    OUT,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    col = tl.arange(0, BD)
    offset = row[:, None] * D + col[None, :]
    mask = (row < ROWS)[:, None] & (col < D)[None, :]
    y = tl.load(Y + offset, mask, other=0).to(tl.float32)
    gate = tl.load(GATE + offset, mask, other=0).to(tl.float32)
    weight = tl.load(W + col, col < D, other=0).to(tl.float32)
    norm = tl.sqrt(tl.sum(y * y, 1) / D + 1e-6)
    out = y / norm[:, None] * weight[None, :]
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(OUT + offset, out * silu, mask)


def norm_gate(y, gate, weight):
    out = torch.empty_like(y)
    d = y.shape[-1]
    rows = y.numel() // d
    _output[(triton.cdiv(rows, 4),)](
        y,
        gate,
        weight,
        out,
        rows,
        d,
        triton.next_power_of_2(d),
        4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
