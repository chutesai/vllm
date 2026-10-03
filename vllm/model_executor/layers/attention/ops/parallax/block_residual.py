# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton forward kernel for Block Attention Residuals."""

from __future__ import annotations

import os
from typing import Any

import torch

try:  # pragma: no cover - import availability depends on runtime.
    from vllm.triton_utils import HAS_TRITON, tl, triton

    if not HAS_TRITON:
        triton = None
        tl = None
except Exception:  # pragma: no cover
    triton = None
    tl = None


MAX_SOURCES = 16
_GRADPROBE_RECORDS: list[tuple[str, torch.Tensor]] = []


def _deterministic_block_attn_res() -> bool:
    """Fixed-order accumulation of the block-attn-res ``grad_qeff``.

    Default OFF, which keeps the shipped ``tl.atomic_add`` reduction. Float
    atomics complete in scheduling order, so the shipped path is not
    reproducible run to run: the step-1 loss is bit-identical (forward only)
    but step 2 differs because step 1's *backward* accumulated in a different
    order. Measured same-box 2-sigma 1.40e-05 nats over 60 steps.

    When ON, every token program stores into its own partial slot and the
    wrapper performs an ordered ``sum(dim=0)``, so the result does not depend
    on block scheduling. Costs ``batch_tokens * d_model * 4`` bytes of scratch
    (18.9 MiB at b1/seq4096/d1152) and one extra reduction pass.

    The OFF path keeps the exact shipped instruction sequence: ``DETERMINISTIC``
    is a ``tl.constexpr``, so the branch is resolved at specialization time and
    the emitted kernel is the pre-port kernel.
    """
    return os.environ.get("MESH_DETERMINISTIC_BLOCK_ATTN_RES", "0") == "1"


def _gradprobe_enabled() -> bool:
    return os.environ.get("MESH_BAR_GRADPROBE", "0") == "1" or (
        os.environ.get("MESH_NANPROBE", "0") == "1"
        and os.environ.get("MESH_NANPROBE_BAR", "1") == "1"
    )


def _record_gradprobe(kind: str, tensors: tuple[torch.Tensor, ...]) -> None:
    if not _gradprobe_enabled() or not tensors:
        return
    counts = torch.stack(
        [(~torch.isfinite(tensor.detach())).sum() for tensor in tensors]
    )
    _GRADPROBE_RECORDS.append((kind, counts))


def reset_block_attn_res_gradprobe() -> None:
    _GRADPROBE_RECORDS.clear()


def consume_block_attn_res_gradprobe() -> list[tuple[str, torch.Tensor]]:
    records = list(_GRADPROBE_RECORDS)
    _GRADPROBE_RECORDS.clear()
    return records


def triton_available() -> bool:
    if os.environ.get("MESH_BAR_TRITON", "1") == "0":
        return False
    return triton is not None and tl is not None and torch.cuda.is_available()


if triton is not None and tl is not None:

    @triton.jit
    def _select_ptr(
        idx: tl.constexpr,
        src0,
        src1,
        src2,
        src3,
        src4,
        src5,
        src6,
        src7,
        src8,
        src9,
        src10,
        src11,
        src12,
        src13,
        src14,
        src15,
    ):
        if idx == 0:
            return src0
        if idx == 1:
            return src1
        if idx == 2:
            return src2
        if idx == 3:
            return src3
        if idx == 4:
            return src4
        if idx == 5:
            return src5
        if idx == 6:
            return src6
        if idx == 7:
            return src7
        if idx == 8:
            return src8
        if idx == 9:
            return src9
        if idx == 10:
            return src10
        if idx == 11:
            return src11
        if idx == 12:
            return src12
        if idx == 13:
            return src13
        if idx == 14:
            return src14
        return src15

    @triton.jit
    def _block_attn_res_forward_kernel(
        q_ptr,
        gamma_ptr,
        out_ptr,
        src0,
        src1,
        src2,
        src3,
        src4,
        src5,
        src6,
        src7,
        src8,
        src9,
        src10,
        src11,
        src12,
        src13,
        src14,
        src15,
        n_tokens: tl.constexpr,
        d_model: tl.constexpr,
        n_sources: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_SOURCE_COUNT: tl.constexpr,
    ):
        token = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < d_model
        q = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        qeff = q * gamma

        max_score = tl.full((), -3.4028234663852886e38, tl.float32)
        for source_idx in tl.static_range(0, MAX_SOURCE_COUNT):
            if source_idx < n_sources:
                src = _select_ptr(
                    source_idx,
                    src0,
                    src1,
                    src2,
                    src3,
                    src4,
                    src5,
                    src6,
                    src7,
                    src8,
                    src9,
                    src10,
                    src11,
                    src12,
                    src13,
                    src14,
                    src15,
                )
                x = tl.load(src + token * d_model + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                mean_sq = tl.sum(x * x, axis=0) / d_model
                inv_rms = tl.rsqrt(mean_sq + eps)
                score = tl.sum(x * qeff, axis=0) * inv_rms
                max_score = tl.maximum(max_score, score)

        denom = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)
        for source_idx in tl.static_range(0, MAX_SOURCE_COUNT):
            if source_idx < n_sources:
                src = _select_ptr(
                    source_idx,
                    src0,
                    src1,
                    src2,
                    src3,
                    src4,
                    src5,
                    src6,
                    src7,
                    src8,
                    src9,
                    src10,
                    src11,
                    src12,
                    src13,
                    src14,
                    src15,
                )
                x = tl.load(src + token * d_model + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                mean_sq = tl.sum(x * x, axis=0) / d_model
                inv_rms = tl.rsqrt(mean_sq + eps)
                score = tl.sum(x * qeff, axis=0) * inv_rms
                weight = tl.exp(score - max_score)
                denom += weight
                acc += weight * x

        y = acc / denom
        tl.store(out_ptr + token * d_model + offs, y, mask=mask)

    @triton.jit
    def _block_attn_res_backward_kernel(
        q_ptr,
        gamma_ptr,
        grad_out_ptr,
        out_ptr,
        grad_qeff_ptr,
        src0,
        src1,
        src2,
        src3,
        src4,
        src5,
        src6,
        src7,
        src8,
        src9,
        src10,
        src11,
        src12,
        src13,
        src14,
        src15,
        grad_src0,
        grad_src1,
        grad_src2,
        grad_src3,
        grad_src4,
        grad_src5,
        grad_src6,
        grad_src7,
        grad_src8,
        grad_src9,
        grad_src10,
        grad_src11,
        grad_src12,
        grad_src13,
        grad_src14,
        grad_src15,
        n_tokens: tl.constexpr,
        d_model: tl.constexpr,
        n_sources: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_SOURCE_COUNT: tl.constexpr,
        DETERMINISTIC: tl.constexpr,
    ):
        token = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < d_model
        q = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        qeff = q * gamma
        grad = tl.load(grad_out_ptr + token * d_model + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        y = tl.load(out_ptr + token * d_model + offs, mask=mask, other=0.0).to(
            tl.float32
        )

        max_score = tl.full((), -3.4028234663852886e38, tl.float32)
        for source_idx in tl.static_range(0, MAX_SOURCE_COUNT):
            if source_idx < n_sources:
                src = _select_ptr(
                    source_idx,
                    src0,
                    src1,
                    src2,
                    src3,
                    src4,
                    src5,
                    src6,
                    src7,
                    src8,
                    src9,
                    src10,
                    src11,
                    src12,
                    src13,
                    src14,
                    src15,
                )
                x = tl.load(src + token * d_model + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                mean_sq = tl.sum(x * x, axis=0) / d_model
                inv_rms = tl.rsqrt(mean_sq + eps)
                score = tl.sum(x * qeff, axis=0) * inv_rms
                max_score = tl.maximum(max_score, score)

        denom = tl.full((), 0.0, tl.float32)
        for source_idx in tl.static_range(0, MAX_SOURCE_COUNT):
            if source_idx < n_sources:
                src = _select_ptr(
                    source_idx,
                    src0,
                    src1,
                    src2,
                    src3,
                    src4,
                    src5,
                    src6,
                    src7,
                    src8,
                    src9,
                    src10,
                    src11,
                    src12,
                    src13,
                    src14,
                    src15,
                )
                x = tl.load(src + token * d_model + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                mean_sq = tl.sum(x * x, axis=0) / d_model
                inv_rms = tl.rsqrt(mean_sq + eps)
                score = tl.sum(x * qeff, axis=0) * inv_rms
                denom += tl.exp(score - max_score)

        grad_qeff = tl.zeros((BLOCK_D,), tl.float32)
        for source_idx in tl.static_range(0, MAX_SOURCE_COUNT):
            if source_idx < n_sources:
                src = _select_ptr(
                    source_idx,
                    src0,
                    src1,
                    src2,
                    src3,
                    src4,
                    src5,
                    src6,
                    src7,
                    src8,
                    src9,
                    src10,
                    src11,
                    src12,
                    src13,
                    src14,
                    src15,
                )
                grad_src = _select_ptr(
                    source_idx,
                    grad_src0,
                    grad_src1,
                    grad_src2,
                    grad_src3,
                    grad_src4,
                    grad_src5,
                    grad_src6,
                    grad_src7,
                    grad_src8,
                    grad_src9,
                    grad_src10,
                    grad_src11,
                    grad_src12,
                    grad_src13,
                    grad_src14,
                    grad_src15,
                )
                x = tl.load(src + token * d_model + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                mean_sq = tl.sum(x * x, axis=0) / d_model
                inv_rms = tl.rsqrt(mean_sq + eps)
                score = tl.sum(x * qeff, axis=0) * inv_rms
                alpha = tl.exp(score - max_score) / denom
                ds = alpha * tl.sum(grad * (x - y), axis=0)
                dscore_dx = inv_rms * qeff - (score * x * inv_rms * inv_rms / d_model)
                grad_x = alpha * grad + ds * dscore_dx
                grad_qeff += ds * inv_rms * x
                tl.store(grad_src + token * d_model + offs, grad_x, mask=mask)

        if DETERMINISTIC:
            # One partial slot per token program. The ordered reduction runs
            # in the wrapper, so accumulation order is independent of block
            # scheduling.
            tl.store(grad_qeff_ptr + token * d_model + offs, grad_qeff, mask=mask)
        else:
            tl.atomic_add(grad_qeff_ptr + offs, grad_qeff, sem="relaxed", mask=mask)

    @triton.jit
    def _block_attn_res_merge_forward_kernel(
        q_ptr,
        gamma_ptr,
        inter_weighted_ptr,
        inter_max_ptr,
        inter_denom_ptr,
        partial_ptr,
        out_ptr,
        n_tokens: tl.constexpr,
        d_model: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < d_model
        q = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        qeff = q * gamma
        weighted = tl.load(
            inter_weighted_ptr + token * d_model + offs, mask=mask, other=0.0
        ).to(tl.float32)
        partial = tl.load(
            partial_ptr + token * d_model + offs, mask=mask, other=0.0
        ).to(tl.float32)
        inter_max = tl.load(inter_max_ptr + token).to(tl.float32)
        inter_denom = tl.load(inter_denom_ptr + token).to(tl.float32)

        mean_sq = tl.sum(partial * partial, axis=0) / d_model
        inv_rms = tl.rsqrt(mean_sq + eps)
        partial_score = tl.sum(partial * qeff, axis=0) * inv_rms
        merged_max = tl.maximum(inter_max, partial_score)
        inter_scale = tl.exp(inter_max - merged_max)
        partial_scale = tl.exp(partial_score - merged_max)
        denom = inter_scale * inter_denom + partial_scale
        out = (weighted * inter_scale + partial * partial_scale) / denom
        tl.store(out_ptr + token * d_model + offs, out, mask=mask)

    @triton.jit
    def _block_attn_res_merge_backward_kernel(
        q_ptr,
        gamma_ptr,
        grad_out_ptr,
        out_ptr,
        inter_weighted_ptr,
        inter_max_ptr,
        inter_denom_ptr,
        partial_ptr,
        grad_qeff_ptr,
        grad_inter_weighted_ptr,
        grad_inter_max_ptr,
        grad_inter_denom_ptr,
        grad_partial_ptr,
        n_tokens: tl.constexpr,
        d_model: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
        DETERMINISTIC: tl.constexpr,
    ):
        token = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < d_model
        q = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        qeff = q * gamma
        grad = tl.load(grad_out_ptr + token * d_model + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        out = tl.load(out_ptr + token * d_model + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        weighted = tl.load(
            inter_weighted_ptr + token * d_model + offs, mask=mask, other=0.0
        ).to(tl.float32)
        partial = tl.load(
            partial_ptr + token * d_model + offs, mask=mask, other=0.0
        ).to(tl.float32)
        inter_max = tl.load(inter_max_ptr + token).to(tl.float32)
        inter_denom = tl.load(inter_denom_ptr + token).to(tl.float32)

        mean_sq = tl.sum(partial * partial, axis=0) / d_model
        inv_rms = tl.rsqrt(mean_sq + eps)
        partial_score = tl.sum(partial * qeff, axis=0) * inv_rms
        merged_max = tl.maximum(inter_max, partial_score)
        inter_scale = tl.exp(inter_max - merged_max)
        partial_scale = tl.exp(partial_score - merged_max)
        denom = inter_scale * inter_denom + partial_scale
        inter_over_denom = inter_scale / denom
        partial_over_denom = partial_scale / denom

        dot_out = tl.sum(grad * out, axis=0)
        dot_inter = tl.sum(grad * (weighted - out * inter_denom), axis=0)
        dot_partial = tl.sum(grad * (partial - out), axis=0)

        grad_weighted = inter_over_denom * grad
        grad_inter_denom = -inter_over_denom * dot_out
        grad_inter_max = inter_over_denom * dot_inter
        grad_partial_score = partial_over_denom * dot_partial
        score_dx = inv_rms * qeff - (
            partial_score * partial * inv_rms * inv_rms / d_model
        )
        grad_partial = partial_over_denom * grad + grad_partial_score * score_dx
        grad_qeff = grad_partial_score * inv_rms * partial

        tl.store(
            grad_inter_weighted_ptr + token * d_model + offs,
            grad_weighted,
            mask=mask,
        )
        tl.store(grad_inter_max_ptr + token, grad_inter_max)
        tl.store(grad_inter_denom_ptr + token, grad_inter_denom)
        tl.store(grad_partial_ptr + token * d_model + offs, grad_partial, mask=mask)
        if DETERMINISTIC:
            tl.store(grad_qeff_ptr + token * d_model + offs, grad_qeff, mask=mask)
        else:
            tl.atomic_add(grad_qeff_ptr + offs, grad_qeff, sem="relaxed", mask=mask)

    @triton.jit
    def _block_attn_res_inter_stats_forward_kernel(
        q_ptr,
        gamma_ptr,
        weighted_ptr,
        max_ptr,
        denom_ptr,
        src0,
        src1,
        src2,
        src3,
        src4,
        src5,
        src6,
        src7,
        src8,
        src9,
        src10,
        src11,
        src12,
        src13,
        src14,
        src15,
        n_tokens: tl.constexpr,
        d_model: tl.constexpr,
        n_blocks: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_BLOCK_COUNT: tl.constexpr,
    ):
        # CUDA grid.y is limited to 65535. Long token axes belong in grid.x.
        token = tl.program_id(0)
        layer_idx = tl.program_id(1)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < d_model
        q = tl.load(q_ptr + layer_idx * d_model + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        gamma = tl.load(
            gamma_ptr + layer_idx * d_model + offs, mask=mask, other=0.0
        ).to(tl.float32)
        qeff = q * gamma

        max_score = tl.full((), -3.4028234663852886e38, tl.float32)
        for block_idx in tl.static_range(0, MAX_BLOCK_COUNT):
            if block_idx < n_blocks:
                src = _select_ptr(
                    block_idx,
                    src0,
                    src1,
                    src2,
                    src3,
                    src4,
                    src5,
                    src6,
                    src7,
                    src8,
                    src9,
                    src10,
                    src11,
                    src12,
                    src13,
                    src14,
                    src15,
                )
                x = tl.load(src + token * d_model + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                mean_sq = tl.sum(x * x, axis=0) / d_model
                inv_rms = tl.rsqrt(mean_sq + eps)
                score = tl.sum(x * qeff, axis=0) * inv_rms
                max_score = tl.maximum(max_score, score)

        denom = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_D,), tl.float32)
        for block_idx in tl.static_range(0, MAX_BLOCK_COUNT):
            if block_idx < n_blocks:
                src = _select_ptr(
                    block_idx,
                    src0,
                    src1,
                    src2,
                    src3,
                    src4,
                    src5,
                    src6,
                    src7,
                    src8,
                    src9,
                    src10,
                    src11,
                    src12,
                    src13,
                    src14,
                    src15,
                )
                x = tl.load(src + token * d_model + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                mean_sq = tl.sum(x * x, axis=0) / d_model
                inv_rms = tl.rsqrt(mean_sq + eps)
                score = tl.sum(x * qeff, axis=0) * inv_rms
                weight = tl.exp(score - max_score)
                denom += weight
                acc += weight * x

        out_offset = (layer_idx * n_tokens + token) * d_model + offs
        tl.store(weighted_ptr + out_offset, acc, mask=mask)
        tl.store(max_ptr + layer_idx * n_tokens + token, max_score)
        tl.store(denom_ptr + layer_idx * n_tokens + token, denom)

    @triton.jit
    def _block_attn_res_inter_stats_backward_kernel(
        q_ptr,
        gamma_ptr,
        grad_weighted_ptr,
        grad_max_ptr,
        grad_denom_ptr,
        saved_max_ptr,
        grad_qeff_ptr,
        src0,
        src1,
        src2,
        src3,
        src4,
        src5,
        src6,
        src7,
        src8,
        src9,
        src10,
        src11,
        src12,
        src13,
        src14,
        src15,
        grad_src0,
        grad_src1,
        grad_src2,
        grad_src3,
        grad_src4,
        grad_src5,
        grad_src6,
        grad_src7,
        grad_src8,
        grad_src9,
        grad_src10,
        grad_src11,
        grad_src12,
        grad_src13,
        grad_src14,
        grad_src15,
        n_tokens: tl.constexpr,
        d_model: tl.constexpr,
        n_blocks: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_BLOCK_COUNT: tl.constexpr,
    ):
        token = tl.program_id(0)
        layer_idx = tl.program_id(1)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < d_model
        q = tl.load(q_ptr + layer_idx * d_model + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        gamma = tl.load(
            gamma_ptr + layer_idx * d_model + offs, mask=mask, other=0.0
        ).to(tl.float32)
        qeff = q * gamma
        grad_weighted = tl.load(
            grad_weighted_ptr + (layer_idx * n_tokens + token) * d_model + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_max = tl.load(grad_max_ptr + layer_idx * n_tokens + token).to(tl.float32)
        grad_denom = tl.load(grad_denom_ptr + layer_idx * n_tokens + token).to(
            tl.float32
        )
        max_score = tl.full((), -3.4028234663852886e38, tl.float32)
        for block_idx in tl.static_range(0, MAX_BLOCK_COUNT):
            if block_idx < n_blocks:
                src = _select_ptr(
                    block_idx,
                    src0,
                    src1,
                    src2,
                    src3,
                    src4,
                    src5,
                    src6,
                    src7,
                    src8,
                    src9,
                    src10,
                    src11,
                    src12,
                    src13,
                    src14,
                    src15,
                )
                x = tl.load(src + token * d_model + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                mean_sq = tl.sum(x * x, axis=0) / d_model
                inv_rms = tl.rsqrt(mean_sq + eps)
                score = tl.sum(x * qeff, axis=0) * inv_rms
                max_score = tl.maximum(max_score, score)

        weighted_dot_sum = tl.full((), 0.0, tl.float32)
        max_count = tl.full((), 0.0, tl.float32)
        for block_idx in tl.static_range(0, MAX_BLOCK_COUNT):
            if block_idx < n_blocks:
                src = _select_ptr(
                    block_idx,
                    src0,
                    src1,
                    src2,
                    src3,
                    src4,
                    src5,
                    src6,
                    src7,
                    src8,
                    src9,
                    src10,
                    src11,
                    src12,
                    src13,
                    src14,
                    src15,
                )
                x = tl.load(src + token * d_model + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                mean_sq = tl.sum(x * x, axis=0) / d_model
                inv_rms = tl.rsqrt(mean_sq + eps)
                score = tl.sum(x * qeff, axis=0) * inv_rms
                weight = tl.exp(score - max_score)
                direct = tl.sum(grad_weighted * x, axis=0) + grad_denom
                weighted_dot_sum += weight * direct
                max_count += tl.where(score == max_score, 1.0, 0.0)

        grad_qeff = tl.zeros((BLOCK_D,), tl.float32)
        for block_idx in tl.static_range(0, MAX_BLOCK_COUNT):
            if block_idx < n_blocks:
                src = _select_ptr(
                    block_idx,
                    src0,
                    src1,
                    src2,
                    src3,
                    src4,
                    src5,
                    src6,
                    src7,
                    src8,
                    src9,
                    src10,
                    src11,
                    src12,
                    src13,
                    src14,
                    src15,
                )
                grad_src = _select_ptr(
                    block_idx,
                    grad_src0,
                    grad_src1,
                    grad_src2,
                    grad_src3,
                    grad_src4,
                    grad_src5,
                    grad_src6,
                    grad_src7,
                    grad_src8,
                    grad_src9,
                    grad_src10,
                    grad_src11,
                    grad_src12,
                    grad_src13,
                    grad_src14,
                    grad_src15,
                )
                x = tl.load(src + token * d_model + offs, mask=mask, other=0.0).to(
                    tl.float32
                )
                mean_sq = tl.sum(x * x, axis=0) / d_model
                inv_rms = tl.rsqrt(mean_sq + eps)
                score = tl.sum(x * qeff, axis=0) * inv_rms
                weight = tl.exp(score - max_score)
                direct = tl.sum(grad_weighted * x, axis=0) + grad_denom
                is_max = score == max_score
                max_share = tl.where(max_count > 0.0, 1.0 / max_count, 0.0)
                dscore = weight * direct + tl.where(
                    is_max, (grad_max - weighted_dot_sum) * max_share, 0.0
                )
                dscore_dx = inv_rms * qeff - (score * x * inv_rms * inv_rms / d_model)
                grad_x = weight * grad_weighted + dscore * dscore_dx
                grad_qeff += dscore * inv_rms * x
                tl.atomic_add(
                    grad_src + token * d_model + offs,
                    grad_x,
                    sem="relaxed",
                    mask=mask,
                )

        tl.atomic_add(
            grad_qeff_ptr + layer_idx * d_model + offs,
            grad_qeff,
            sem="relaxed",
            mask=mask,
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _num_warps_for_block(block_d: int) -> int:
    if block_d >= 2048:
        return 8
    if block_d >= 512:
        return 4
    return 1


def _source_grads(
    grad_out: torch.Tensor,
    out: torch.Tensor,
    q: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    sources: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    grad = grad_out.float()
    y = out.float()
    q_f = q.float()
    gamma_f = gamma.float()
    qeff = q_f * gamma_f
    d_model = sources[0].shape[-1]

    scores: list[torch.Tensor] = []
    invs: list[torch.Tensor] = []
    xs: list[torch.Tensor] = []
    for source in sources:
        x = source.float()
        inv = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        score = (x * qeff).sum(dim=-1) * inv.squeeze(-1)
        xs.append(x)
        invs.append(inv)
        scores.append(score)

    alpha = torch.stack(scores, dim=0).softmax(dim=0)
    grad_qeff = torch.zeros_like(qeff)
    grad_sources: list[torch.Tensor] = []
    for idx, (x, inv, source) in enumerate(zip(xs, invs, sources)):
        a = alpha[idx].unsqueeze(-1)
        ds = alpha[idx].unsqueeze(-1) * (grad * (x - y)).sum(dim=-1, keepdim=True)
        score = scores[idx].unsqueeze(-1)
        dscore_dx = inv * qeff - (score * x * inv.square() / float(d_model))
        grad_x = a * grad + ds * dscore_dx
        grad_qeff = grad_qeff + (ds * inv * x).sum(dim=tuple(range(x.ndim - 1)))
        grad_sources.append(grad_x.to(source.dtype))

    return (
        (grad_qeff * gamma_f).to(q.dtype),
        (grad_qeff * q_f).to(gamma.dtype),
        tuple(grad_sources),
    )


class _TritonBlockAttnResFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        *sources: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not triton_available()
            or not sources
            or len(sources) > MAX_SOURCES
            or not all(source.is_cuda for source in sources)
        ):
            raise RuntimeError("Triton BlockAttnRes forward requires CUDA tensors")
        if any(source.shape != sources[0].shape for source in sources):
            raise ValueError("All BlockAttnRes sources must have the same shape")
        if any(not source.is_contiguous() for source in sources):
            sources = tuple(source.contiguous() for source in sources)

        batch_tokens = sources[0].numel() // sources[0].shape[-1]
        d_model = int(sources[0].shape[-1])
        block_d = _next_power_of_2(d_model)
        num_warps = _num_warps_for_block(block_d)
        if block_d > 8192:
            raise RuntimeError("Triton BlockAttnRes supports d_model <= 8192")
        out = torch.empty_like(sources[0])
        ptrs = list(sources) + [sources[0]] * (MAX_SOURCES - len(sources))
        assert triton is not None
        _block_attn_res_forward_kernel[(batch_tokens,)](
            q,
            gamma,
            out,
            *ptrs,
            batch_tokens,
            d_model,
            len(sources),
            float(eps),
            BLOCK_D=block_d,
            MAX_SOURCE_COUNT=MAX_SOURCES,
            num_warps=num_warps,
        )
        ctx.eps = float(eps)
        ctx.save_for_backward(q, gamma, out, *sources)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        q, gamma, out, *sources = ctx.saved_tensors
        if (
            triton_available()
            and grad_out.is_cuda
            and out.is_cuda
            and all(source.is_cuda for source in sources)
        ):
            grad_out_c = grad_out.contiguous()
            grad_sources = tuple(torch.empty_like(source) for source in sources)
            batch_tokens = sources[0].numel() // sources[0].shape[-1]
            d_model = int(sources[0].shape[-1])
            block_d = _next_power_of_2(d_model)
            num_warps = _num_warps_for_block(block_d)
            deterministic_qeff = _deterministic_block_attn_res()
            grad_qeff = torch.zeros(
                (batch_tokens, d_model) if deterministic_qeff else (d_model,),
                device=sources[0].device,
                dtype=torch.float32,
            )
            ptrs = list(sources) + [sources[0]] * (MAX_SOURCES - len(sources))
            # Never alias inactive pointer slots to a live source gradient.
            # The slots are masked by the constexpr source count, but a bad
            # specialization must not be able to overwrite source zero.
            if os.environ.get("MESH_BAR_SAFE_PADDING", "1") == "1":
                grad_padding = torch.empty_like(sources[0])
            else:
                grad_padding = grad_sources[0]
            grad_ptrs = list(grad_sources) + [grad_padding] * (
                MAX_SOURCES - len(grad_sources)
            )
            assert triton is not None
            _block_attn_res_backward_kernel[(batch_tokens,)](
                q,
                gamma,
                grad_out_c,
                out,
                grad_qeff,
                *ptrs,
                *grad_ptrs,
                batch_tokens,
                d_model,
                len(sources),
                float(ctx.eps),
                BLOCK_D=block_d,
                MAX_SOURCE_COUNT=MAX_SOURCES,
                DETERMINISTIC=deterministic_qeff,
                num_warps=num_warps,
            )
            if deterministic_qeff:
                grad_qeff = grad_qeff.sum(dim=0)
            _record_gradprobe("head_sources", grad_sources)
            return (
                (grad_qeff * gamma.float()).to(q.dtype),
                (grad_qeff * q.float()).to(gamma.dtype),
                None,
                *grad_sources,
            )
        grad_q, grad_gamma, grad_sources = _source_grads(
            grad_out, out, q, gamma, ctx.eps, tuple(sources)
        )
        return grad_q, grad_gamma, None, *grad_sources


def fused_block_attn_res(
    q: torch.Tensor,
    gamma: torch.Tensor,
    sources: list[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    return _TritonBlockAttnResFn.apply(q, gamma, float(eps), *sources)


class _TritonBlockAttnResMergeFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        gamma: torch.Tensor,
        inter_weighted: torch.Tensor,
        inter_max: torch.Tensor,
        inter_denom: torch.Tensor,
        partial: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        if not triton_available() or not partial.is_cuda:
            raise RuntimeError("Triton BlockAttnRes merge requires CUDA tensors")
        inter_weighted = inter_weighted.contiguous()
        inter_max = inter_max.contiguous()
        inter_denom = inter_denom.contiguous()
        partial = partial.contiguous()
        out = torch.empty_like(partial)
        batch_tokens = partial.numel() // partial.shape[-1]
        d_model = int(partial.shape[-1])
        block_d = _next_power_of_2(d_model)
        num_warps = _num_warps_for_block(block_d)
        if block_d > 8192:
            raise RuntimeError("Triton BlockAttnRes merge supports d_model <= 8192")
        assert triton is not None
        _block_attn_res_merge_forward_kernel[(batch_tokens,)](
            q,
            gamma,
            inter_weighted,
            inter_max,
            inter_denom,
            partial,
            out,
            batch_tokens,
            d_model,
            float(eps),
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        ctx.eps = float(eps)
        ctx.save_for_backward(
            q, gamma, inter_weighted, inter_max, inter_denom, partial, out
        )
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[override]
        q, gamma, inter_weighted, inter_max, inter_denom, partial, out = (
            ctx.saved_tensors
        )
        if triton_available() and grad_out.is_cuda:
            grad_out_c = grad_out.contiguous()
            batch_tokens = partial.numel() // partial.shape[-1]
            d_model = int(partial.shape[-1])
            block_d = _next_power_of_2(d_model)
            num_warps = _num_warps_for_block(block_d)
            deterministic_qeff = _deterministic_block_attn_res()
            grad_qeff = torch.zeros(
                (batch_tokens, d_model) if deterministic_qeff else (d_model,),
                device=partial.device,
                dtype=torch.float32,
            )
            grad_inter_weighted = torch.empty_like(inter_weighted)
            grad_inter_max = torch.empty_like(inter_max)
            grad_inter_denom = torch.empty_like(inter_denom)
            grad_partial = torch.empty_like(partial)
            assert triton is not None
            _block_attn_res_merge_backward_kernel[(batch_tokens,)](
                q,
                gamma,
                grad_out_c,
                out,
                inter_weighted,
                inter_max,
                inter_denom,
                partial,
                grad_qeff,
                grad_inter_weighted,
                grad_inter_max,
                grad_inter_denom,
                grad_partial,
                batch_tokens,
                d_model,
                float(ctx.eps),
                BLOCK_D=block_d,
                DETERMINISTIC=deterministic_qeff,
                num_warps=num_warps,
            )
            if deterministic_qeff:
                grad_qeff = grad_qeff.sum(dim=0)
            _record_gradprobe(
                "merge",
                (grad_inter_weighted, grad_inter_max, grad_inter_denom, grad_partial),
            )
            return (
                (grad_qeff * gamma.float()).to(q.dtype),
                (grad_qeff * q.float()).to(gamma.dtype),
                grad_inter_weighted,
                grad_inter_max,
                grad_inter_denom,
                grad_partial,
                None,
            )
        raise RuntimeError("Triton BlockAttnRes merge backward requires CUDA tensors")


def fused_block_attn_res_merge(
    q: torch.Tensor,
    gamma: torch.Tensor,
    inter_weighted: torch.Tensor,
    inter_max: torch.Tensor,
    inter_denom: torch.Tensor,
    partial: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _TritonBlockAttnResMergeFn.apply(
        q, gamma, inter_weighted, inter_max, inter_denom, partial, float(eps)
    )


class _TritonBlockAttnResInterStatsFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        *blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            not triton_available()
            or not blocks
            or len(blocks) > MAX_SOURCES
            or not all(block.is_cuda for block in blocks)
        ):
            raise RuntimeError("Triton BlockAttnRes inter-stats requires CUDA tensors")
        if any(block.shape != blocks[0].shape for block in blocks):
            raise ValueError("All BlockAttnRes blocks must have the same shape")
        if any(not block.is_contiguous() for block in blocks):
            blocks = tuple(block.contiguous() for block in blocks)
        q = q.contiguous()
        gamma = gamma.contiguous()
        n_layers = int(q.shape[0])
        batch_tokens = blocks[0].numel() // blocks[0].shape[-1]
        d_model = int(blocks[0].shape[-1])
        block_d = _next_power_of_2(d_model)
        num_warps = _num_warps_for_block(block_d)
        if block_d > 8192:
            raise RuntimeError(
                "Triton BlockAttnRes inter-stats supports d_model <= 8192"
            )
        weighted = torch.empty(
            n_layers,
            *blocks[0].shape,
            device=blocks[0].device,
            dtype=blocks[0].dtype,
        )
        max_score = torch.empty(
            n_layers,
            *blocks[0].shape[:-1],
            device=blocks[0].device,
            dtype=torch.float32,
        )
        denom = torch.empty_like(max_score)
        ptrs = list(blocks) + [blocks[0]] * (MAX_SOURCES - len(blocks))
        assert triton is not None
        _block_attn_res_inter_stats_forward_kernel[(batch_tokens, n_layers)](
            q,
            gamma,
            weighted,
            max_score,
            denom,
            *ptrs,
            batch_tokens,
            d_model,
            len(blocks),
            float(eps),
            BLOCK_D=block_d,
            MAX_BLOCK_COUNT=MAX_SOURCES,
            num_warps=num_warps,
        )
        ctx.eps = float(eps)
        ctx.save_for_backward(q, gamma, max_score, *blocks)
        return weighted, max_score, denom

    @staticmethod
    def backward(ctx: Any, grad_weighted, grad_max, grad_denom):  # type: ignore[override]
        q, gamma, max_score, *blocks = ctx.saved_tensors
        if (
            not triton_available()
            or grad_weighted is None
            or grad_max is None
            or grad_denom is None
        ):
            raise RuntimeError("Triton BlockAttnRes inter-stats backward requires CUDA")
        grad_weighted = grad_weighted.contiguous()
        grad_max = grad_max.contiguous()
        grad_denom = grad_denom.contiguous()
        n_layers = int(q.shape[0])
        batch_tokens = blocks[0].numel() // blocks[0].shape[-1]
        d_model = int(blocks[0].shape[-1])
        block_d = _next_power_of_2(d_model)
        num_warps = _num_warps_for_block(block_d)
        grad_qeff = torch.zeros(
            n_layers, d_model, device=blocks[0].device, dtype=torch.float32
        )
        # Every target layer contributes to the same completed-block source.
        # FP32 accumulation is available as a higher-precision diagnostic,
        # while bf16 remains the throughput default after padding is isolated.
        grad_accum_dtype = (
            torch.float32
            if os.environ.get("MESH_BAR_FP32_GRAD_ACCUM", "0") == "1"
            else blocks[0].dtype
        )
        grad_blocks_accum = tuple(
            torch.zeros_like(block, dtype=grad_accum_dtype) for block in blocks
        )
        ptrs = list(blocks) + [blocks[0]] * (MAX_SOURCES - len(blocks))
        if os.environ.get("MESH_BAR_SAFE_PADDING", "1") == "1":
            grad_padding = torch.zeros_like(blocks[0], dtype=grad_accum_dtype)
        else:
            grad_padding = grad_blocks_accum[0]
        grad_ptrs = list(grad_blocks_accum) + [grad_padding] * (
            MAX_SOURCES - len(grad_blocks_accum)
        )
        assert triton is not None
        _block_attn_res_inter_stats_backward_kernel[(batch_tokens, n_layers)](
            q,
            gamma,
            grad_weighted,
            grad_max,
            grad_denom,
            max_score,
            grad_qeff,
            *ptrs,
            *grad_ptrs,
            batch_tokens,
            d_model,
            len(blocks),
            float(ctx.eps),
            BLOCK_D=block_d,
            MAX_BLOCK_COUNT=MAX_SOURCES,
            num_warps=num_warps,
        )
        grad_blocks = tuple(
            grad.to(block.dtype) for grad, block in zip(grad_blocks_accum, blocks)
        )
        _record_gradprobe("inter_blocks", grad_blocks)
        grad_q = (grad_qeff * gamma.float()).to(q.dtype)
        grad_gamma = (grad_qeff * q.float()).to(gamma.dtype)
        return grad_q, grad_gamma, None, *grad_blocks


class _TritonBlockAttnResInterStatsSplitFn(torch.autograd.Function):
    """Inter-block stats with one autograd output per target layer.

    A single stacked output followed by ``select`` views makes autograd scatter
    every layer gradient into a fresh full ``[L,B,T,D]`` tensor and repeatedly
    add those tensors. Returning independent outputs lets backward assemble
    the stack exactly once before the existing fused kernel.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        *blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if (
            not triton_available()
            or not blocks
            or len(blocks) > MAX_SOURCES
            or not all(block.is_cuda for block in blocks)
        ):
            raise RuntimeError(
                "Triton BlockAttnRes split inter-stats requires CUDA tensors"
            )
        if any(block.shape != blocks[0].shape for block in blocks):
            raise ValueError("All BlockAttnRes blocks must have the same shape")
        blocks = tuple(block.contiguous() for block in blocks)
        q = q.contiguous()
        gamma = gamma.contiguous()
        n_layers = int(q.shape[0])
        batch_tokens = blocks[0].numel() // blocks[0].shape[-1]
        d_model = int(blocks[0].shape[-1])
        block_d = _next_power_of_2(d_model)
        if block_d > 8192:
            raise RuntimeError(
                "Triton BlockAttnRes split inter-stats supports d_model <= 8192"
            )
        weighted = torch.empty(
            n_layers,
            *blocks[0].shape,
            device=blocks[0].device,
            dtype=blocks[0].dtype,
        )
        max_score = torch.empty(
            n_layers,
            *blocks[0].shape[:-1],
            device=blocks[0].device,
            dtype=torch.float32,
        )
        denom = torch.empty_like(max_score)
        ptrs = list(blocks) + [blocks[0]] * (MAX_SOURCES - len(blocks))
        assert triton is not None
        _block_attn_res_inter_stats_forward_kernel[(batch_tokens, n_layers)](
            q,
            gamma,
            weighted,
            max_score,
            denom,
            *ptrs,
            batch_tokens,
            d_model,
            len(blocks),
            float(eps),
            BLOCK_D=block_d,
            MAX_BLOCK_COUNT=MAX_SOURCES,
            num_warps=_num_warps_for_block(block_d),
        )
        ctx.eps = float(eps)
        ctx.n_layers = n_layers
        ctx.save_for_backward(q, gamma, max_score, *blocks)
        return (
            *weighted.unbind(0),
            *max_score.unbind(0),
            *denom.unbind(0),
        )

    @staticmethod
    def backward(ctx: Any, *output_grads):  # type: ignore[override]
        q, gamma, max_score, *blocks = ctx.saved_tensors
        n_layers = int(ctx.n_layers)
        if not triton_available() or len(output_grads) != 3 * n_layers:
            raise RuntimeError(
                "Triton BlockAttnRes split inter-stats backward requires CUDA"
            )
        weighted_template = blocks[0]
        scalar_template = max_score[0]

        def _stack_or_zero(grads, template):
            return torch.stack(
                [
                    torch.zeros_like(template) if grad is None else grad
                    for grad in grads
                ],
                dim=0,
            ).contiguous()

        grad_weighted = _stack_or_zero(output_grads[:n_layers], weighted_template)
        grad_max = _stack_or_zero(
            output_grads[n_layers : 2 * n_layers], scalar_template
        )
        grad_denom = _stack_or_zero(output_grads[2 * n_layers :], scalar_template)
        batch_tokens = blocks[0].numel() // blocks[0].shape[-1]
        d_model = int(blocks[0].shape[-1])
        block_d = _next_power_of_2(d_model)
        grad_qeff = torch.zeros(
            n_layers,
            d_model,
            device=blocks[0].device,
            dtype=torch.float32,
        )
        grad_accum_dtype = (
            torch.float32
            if os.environ.get("MESH_BAR_FP32_GRAD_ACCUM", "0") == "1"
            else blocks[0].dtype
        )
        grad_blocks_accum = tuple(
            torch.zeros_like(block, dtype=grad_accum_dtype) for block in blocks
        )
        ptrs = list(blocks) + [blocks[0]] * (MAX_SOURCES - len(blocks))
        if os.environ.get("MESH_BAR_SAFE_PADDING", "1") == "1":
            grad_padding = torch.zeros_like(blocks[0], dtype=grad_accum_dtype)
        else:
            grad_padding = grad_blocks_accum[0]
        grad_ptrs = list(grad_blocks_accum) + [grad_padding] * (
            MAX_SOURCES - len(grad_blocks_accum)
        )
        assert triton is not None
        _block_attn_res_inter_stats_backward_kernel[(batch_tokens, n_layers)](
            q,
            gamma,
            grad_weighted,
            grad_max,
            grad_denom,
            max_score,
            grad_qeff,
            *ptrs,
            *grad_ptrs,
            batch_tokens,
            d_model,
            len(blocks),
            float(ctx.eps),
            BLOCK_D=block_d,
            MAX_BLOCK_COUNT=MAX_SOURCES,
            num_warps=_num_warps_for_block(block_d),
        )
        grad_blocks = tuple(
            grad.to(block.dtype) for grad, block in zip(grad_blocks_accum, blocks)
        )
        _record_gradprobe("inter_split_blocks", grad_blocks)
        grad_q = (grad_qeff * gamma.float()).to(q.dtype)
        grad_gamma = (grad_qeff * q.float()).to(gamma.dtype)
        return grad_q, grad_gamma, None, *grad_blocks


def fused_batched_inter_stats(
    q: torch.Tensor,
    gamma: torch.Tensor,
    blocks: list[torch.Tensor],
    eps: float,
) -> tuple[Any, Any, Any]:
    if os.environ.get("MESH_BAR_SPLIT_OUTPUTS", "0") == "1":
        flat = _TritonBlockAttnResInterStatsSplitFn.apply(q, gamma, float(eps), *blocks)
        n_layers = int(q.shape[0])
        return (
            tuple(flat[:n_layers]),
            tuple(flat[n_layers : 2 * n_layers]),
            tuple(flat[2 * n_layers :]),
        )
    return _TritonBlockAttnResInterStatsFn.apply(q, gamma, float(eps), *blocks)
