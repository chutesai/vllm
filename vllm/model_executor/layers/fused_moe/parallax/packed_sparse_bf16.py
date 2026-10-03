# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental lossless paired-4:8 -> native BF16 2:4 sparse MMA.

Permuting each activation octet to even columns followed by odd columns
turns two retained pairs into two retained positions in each quartet.
Weights retain their BF16 row scale; activations are never requantized.
"""

import torch

from vllm.tilelang_utils import T, tilelang
from vllm.triton_utils import tl, triton

from ..moe_align_block_size import (
    moe_align_block_size,
)
from .ternary_decode import _combine, pack


def pack_sparse(codes):
    if codes.ndim != 3 or codes.dtype != torch.int8 or codes.shape[-1] % 16:
        raise ValueError("3D int8 ternary codes with K divisible by 16 required")
    experts, n, k = codes.shape
    perm = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=codes.device)
    groups = (
        codes.reshape(experts, n, k // 8, 8)
        .index_select(-1, perm)
        .reshape(experts, n, k // 4, 4)
    )
    nz = groups != 0
    pair_present = nz.reshape(experts, n, k // 8, 2, 4).any(-2)
    if bool((pair_present.sum(-1) > 2).any()):
        raise ValueError("Weights violate paired-4:8 sparsity")
    positions = torch.arange(4, device=codes.device)
    selected = (
        torch.where(pair_present, positions, positions + 4)
        .argsort(-1)[..., :2]
        .sort(-1)
        .values
    )
    selected = (
        selected[..., None, :]
        .expand(experts, n, k // 8, 2, 2)
        .reshape(experts, n, k // 4, 2)
    )
    values = groups.gather(-1, selected).reshape(experts, n, k // 2).contiguous()
    nibble = selected[..., 0] | (selected[..., 1] << 2)
    chunks = nibble.reshape(experts, n, k // 16, 4)
    meta = sum(chunks[..., i] << (4 * i) for i in range(4)).to(torch.int16).contiguous()
    # Verify reconstruction, including zero-filled retained positions.
    reconstructed = torch.zeros_like(groups).scatter_(
        -1, selected, groups.gather(-1, selected)
    )
    torch.testing.assert_close(reconstructed, groups, atol=0, rtol=0)
    return pack(values), meta


def expand_sparse(packed, alpha):
    """Materialize the retained BF16 values exactly, keeping sparse metadata."""
    if (
        packed.ndim != 3
        or packed.dtype != torch.uint8
        or alpha.dtype != torch.bfloat16
        or alpha.shape != packed.shape[:2]
        or alpha.device != packed.device
    ):
        raise ValueError(
            "Packed uint8 expert bank and matching BF16 row scales required"
        )
    shifts = torch.arange(4, device=packed.device, dtype=torch.uint8) * 2
    code = ((packed[..., None] >> shifts) & 3).flatten(-2).to(torch.int8)
    signed = (1 - (code & 2)) * (code != 0).to(torch.int8)
    return (signed.to(alpha.dtype) * alpha[..., None]).contiguous()


@triton.jit
def _gather(
    X,
    SORTED,
    Y,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOP: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    kk = tl.program_id(1) * B + tl.arange(0, B)
    original = kk // 8 * 8 + tl.where(kk % 8 < 4, kk % 8 * 2, (kk % 8 - 4) * 2 + 1)
    route = tl.load(SORTED + row)
    value = tl.load(X + route // TOP * K + original, (route < ROWS) & (kk < K), other=0)
    tl.store(Y + row * K + kk, value, kk < K)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def grouped_kernel(
    ROWS,
    E,
    N,
    K,
    TOP,
    BLOCKS,
    SORTED_SIZE,
    FIRST,
    BM=32,
    BN=64,
    BK=64,
    STAGES=2,
    THREADS=128,
    PREEXPANDED=False,
):
    @T.prim_func
    def main(
        X: T.Tensor((SORTED_SIZE, K), T.bfloat16),  # type: ignore[valid-type]
        W: T.Tensor(  # type: ignore[valid-type]
            (E, N, K // (2 if PREEXPANDED else 8)),
            T.bfloat16 if PREEXPANDED else T.uint8,
        ),
        META: T.Tensor((E, N, K // 16), T.int16),  # type: ignore[valid-type]
        ALPHA: T.Tensor((E, N), T.bfloat16),  # type: ignore[valid-type]
        GATES: T.Tensor((ROWS,), T.float32),  # type: ignore[valid-type]
        SORTED: T.Tensor((SORTED_SIZE,), T.int32),  # type: ignore[valid-type]
        EXPERTS: T.Tensor((BLOCKS,), T.int32),  # type: ignore[valid-type]
        COUNT: T.Tensor((1,), T.int32),  # type: ignore[valid-type]
        Y: T.Tensor(  # type: ignore[valid-type]
            (SORTED_SIZE if FIRST else ROWS, N), T.bfloat16 if FIRST else T.float32
        ),
    ):
        with T.Kernel(T.ceildiv(N, BN), BLOCKS, threads=THREADS) as (bx, by):
            A_shared = T.alloc_shared((BN, BK // 2), T.bfloat16)
            E_shared = T.alloc_shared((BN, BK // 16), T.int16)
            B_shared = T.alloc_shared((BM, BK), T.bfloat16)
            C_local = T.alloc_fragment((BN, BM), T.float32)
            if by * BM < COUNT[0]:
                expert = EXPERTS[by]
                T.clear(C_local)
                for block in T.Pipelined(T.ceildiv(K, BK), num_stages=STAGES):
                    if PREEXPANDED:
                        T.copy(W[expert, bx * BN, block * BK // 2], A_shared)
                    else:
                        for i, j in T.Parallel(BN, BK // 8):
                            packed_byte = W[
                                expert, bx * BN + i, block * BK // 8 + j
                            ].astype(T.int32)
                            scale = ALPHA[expert, bx * BN + i].astype(T.float32)
                            for v in T.unroll(4):
                                code = (packed_byte >> (v * 2)) & 3
                                signed = (1 - (code & 2)) * (code != 0).astype(T.int32)
                                A_shared[i, j * 4 + v] = scale * signed
                    T.copy(META[expert, bx * BN, block * BK // 16], E_shared)
                    T.copy(X[by * BM, block * BK], B_shared)
                    T.gemm_sp(A_shared, E_shared, B_shared, C_local, transpose_B=True)
                for i, j in T.Parallel(BN, BM):
                    route = SORTED[by * BM + j]
                    if (FIRST or route < ROWS) and bx * BN + i < N:
                        value = C_local[i, j].astype(T.bfloat16).astype(T.float32)
                        if FIRST:
                            positive = T.max(value, 0)
                            nn = bx * BN + i
                            permuted_n = nn // 8 * 8 + nn % 8 // 2 + (nn % 2) * 4
                            Y[by * BM + j, permuted_n] = positive * positive
                        else:
                            Y[route, bx * BN + i] = value * GATES[route]

    return main


def moe(
    x,
    up,
    down,
    alpha_up,
    alpha_down,
    meta_up,
    meta_down,
    ids,
    gates,
    block_m=32,
    block_n=64,
    block_k=64,
    stages=2,
    threads=128,
    preexpanded=False,
):
    tokens, latent = x.shape
    experts, hidden, _ = up.shape
    top = ids.shape[1]
    ids = ids.to(torch.int32)
    sorted_ids, expert_ids, count = moe_align_block_size(
        ids, block_m, experts, pad_sorted_ids=True
    )
    gathered = torch.empty((sorted_ids.numel(), latent), device=x.device, dtype=x.dtype)
    _gather[(sorted_ids.numel(), triton.cdiv(latent, 128))](
        x, sorted_ids, gathered, tokens * top, latent, top, 128
    )
    h = torch.empty((sorted_ids.numel(), hidden), device=x.device, dtype=x.dtype)
    partial = torch.empty((tokens * top, latent), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    for first, src, weight, alpha, meta, dst, n, k in (
        (True, gathered, up, alpha_up, meta_up, h, hidden, latent),
        (False, h, down, alpha_down, meta_down, partial, latent, hidden),
    ):
        kernel = grouped_kernel(
            tokens * top,
            experts,
            n,
            k,
            top,
            expert_ids.numel(),
            sorted_ids.numel(),
            first,
            block_m,
            block_n,
            block_k,
            stages,
            threads,
            preexpanded,
        )
        kernel(
            src,
            weight,
            meta,
            alpha,
            gates.reshape(-1),
            sorted_ids,
            expert_ids,
            count,
            dst,
        )
    _combine[(tokens, triton.cdiv(latent, 128))](
        partial, y, latent, top, 128, triton.next_power_of_2(top)
    )
    return y
