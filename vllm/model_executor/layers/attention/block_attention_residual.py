# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Block Attention Residuals (Kimi K2.6, arXiv:2603.15031).

Learned pseudo-query attends over completed block representations at
attention boundary positions. Zero-initialized projection ensures
uniform averaging at initialization (no disruption to pretrained weights).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from vllm.model_executor.layers.attention.block_residual_norm import gain_for


class BlockAttnRes(nn.Module):
    """Single block attention residual aggregation point.

    Learned pseudo-query attends over completed block representations.
    Zero-initialized projection weight → uniform softmax at t=0.

    Args:
        d_model: Feature dimension.
        norm_eps: RMSNorm epsilon.

    """

    def __init__(self, d_model: int, norm_eps: float = 1e-6) -> None:
        super().__init__()
        from vllm.model_executor.layers.attention.block_residual_norm import RMSNorm

        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.proj = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def _source_logits(self, source: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(source)).squeeze(-1)

    def attend_stats(
        self, sources: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unnormalized online-softmax stats over depth sources."""
        if not sources:
            raise ValueError("BlockAttnRes requires at least one source tensor")
        logits = [self._source_logits(source) for source in sources]
        max_logit = torch.stack(logits, dim=0).amax(dim=0)
        denom = torch.zeros_like(max_logit)
        weighted = torch.zeros_like(sources[0])
        for logit, source in zip(logits, sources):
            weight = torch.exp(logit - max_logit)
            denom.add_(weight)
            weighted.add_(source * weight.unsqueeze(-1).to(source.dtype))
        return weighted, max_logit, denom

    def stats_to_output(
        self, weighted: torch.Tensor, denom: torch.Tensor
    ) -> torch.Tensor:
        return weighted / denom.clamp_min(self.norm.eps).unsqueeze(-1).to(
            weighted.dtype
        )

    def merge_inter_stats_with_partial(
        self,
        inter_weighted: torch.Tensor,
        inter_max: torch.Tensor,
        inter_denom: torch.Tensor,
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        """Online-softmax merge of cached inter-block stats with partial block."""
        if partial_block.is_cuda:
            from .ops.parallax.block_residual import (
                fused_block_attn_res_merge,
                triton_available,
            )

            if triton_available():
                return fused_block_attn_res_merge(
                    self.proj.weight.squeeze(0),
                    gain_for(self.norm.weight, partial_block.dtype),
                    inter_weighted,
                    inter_max,
                    inter_denom,
                    partial_block,
                    self.norm.eps,
                )

        partial_logit = self._source_logits(partial_block)
        merged_max = torch.maximum(inter_max, partial_logit)
        inter_scale = torch.exp(inter_max - merged_max)
        partial_scale = torch.exp(partial_logit - merged_max)
        denom = inter_scale * inter_denom + partial_scale
        weighted = inter_weighted * inter_scale.unsqueeze(-1).to(
            inter_weighted.dtype
        ) + partial_block * partial_scale.unsqueeze(-1).to(partial_block.dtype)
        return self.stats_to_output(weighted, denom)

    @staticmethod
    def batched_inter_stats(
        modules: Sequence[BlockAttnRes],
        blocks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Phase-1 inter-block attention for all layers in one block.

        This batches the learned pseudo-queries for the layers in the current
        block against the already completed block representations.  It is exact
        for RMSNorm-with-weight because the learned norm weight can be absorbed
        into the pseudo-query vector.
        """
        if not modules:
            raise ValueError("batched_inter_stats requires at least one module")
        if not blocks:
            raise ValueError("batched_inter_stats requires at least one block")
        first = modules[0]
        eps = first.norm.eps
        for module in modules[1:]:
            if module.norm.eps != eps:
                # Exact fallback for unusual mixed-epsilon configurations.
                stats = [module.attend_stats(blocks) for module in modules]
                return (
                    torch.stack([item[0] for item in stats], dim=0),
                    torch.stack([item[1] for item in stats], dim=0),
                    torch.stack([item[2] for item in stats], dim=0),
                )

        if blocks[0].is_cuda and len(blocks) <= 16 and len(modules) <= 16:
            from .ops.parallax.block_residual import (
                fused_batched_inter_stats,
                triton_available,
            )

            if triton_available():
                gamma = torch.stack(
                    [
                        gain_for(module.norm.weight, blocks[0].dtype)
                        for module in modules
                    ],
                    dim=0,
                )
                q = torch.stack(
                    [module.proj.weight.squeeze(0) for module in modules],
                    dim=0,
                )
                return fused_batched_inter_stats(
                    q,
                    gamma,
                    [block.contiguous() for block in blocks],
                    eps,
                )

        queries = torch.stack(
            [
                module.proj.weight.squeeze(0).float()
                * module.norm.weight.detach().float()
                if not module.norm.weight.requires_grad
                else module.proj.weight.squeeze(0).float() * module.norm.weight.float()
                for module in modules
            ],
            dim=0,
        )
        max_logit: torch.Tensor | None = None
        logits_by_block: list[torch.Tensor] = []
        for block in blocks:
            variance = block.float().pow(2).mean(dim=-1, keepdim=True)
            normed = block.float() * torch.rsqrt(variance + eps)
            logits = torch.einsum("ld,btd->lbt", queries, normed)
            logits_by_block.append(logits)
            max_logit = (
                logits if max_logit is None else torch.maximum(max_logit, logits)
            )
        assert max_logit is not None

        denom = torch.zeros_like(max_logit)
        weighted = torch.zeros(
            len(modules),
            *blocks[0].shape,
            device=blocks[0].device,
            dtype=blocks[0].dtype,
        )
        for logits, block in zip(logits_by_block, blocks):
            weight = torch.exp(logits - max_logit)
            denom.add_(weight)
            weighted.add_(block.unsqueeze(0) * weight.unsqueeze(-1).to(block.dtype))
        return weighted, max_logit, denom

    def forward(
        self, blocks: list[torch.Tensor], partial_block: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Aggregate block representations via learned softmax attention.

        Args:
            blocks: List of completed block tensors [B, T, D].
            partial_block: Current in-progress block [B, T, D], omitted for the
                first layer in a block.

        Returns:
            Aggregated representation [B, T, D].

        """
        sources = blocks if partial_block is None else [*blocks, partial_block]
        if not sources:
            raise ValueError("BlockAttnRes requires at least one source tensor")
        if len(sources) == 1:
            return sources[0]
        if sources[0].is_cuda and len(sources) <= 16:
            from .ops.parallax.block_residual import (
                fused_block_attn_res,
                triton_available,
            )

            if triton_available():
                return fused_block_attn_res(
                    self.proj.weight.squeeze(0),
                    gain_for(self.norm.weight, sources[0].dtype),
                    [source.contiguous() for source in sources],
                    self.norm.eps,
                )

        weighted, _max_logit, denom = self.attend_stats(sources)
        return self.stats_to_output(weighted, denom)
