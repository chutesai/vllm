# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in V2 sampler optimization: reuse FP32 logits when raw scores are unobserved.

The operation order, temperature, filters and RNG are unchanged. Requests for
logprobs and non-FP32 input retain the original copy. No installed vLLM files
are modified; only the experimental worker's existing sampler is specialized.
"""

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from vllm.v1.core.sched.parallax_decode import DecodeSlotCacheWorker
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext
from vllm.v1.worker.gpu.sample.sampler import Sampler

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class InplaceSampler(Sampler):
    _reuse_logits: bool
    reused_logits_calls: int

    def sample(self, *args, **kwargs):
        return_logprobs = kwargs.get(
            "return_logprobs", args[8] if len(args) > 8 else False
        )
        self._reuse_logits = not return_logprobs and not self.return_sampling_mask
        try:
            return super().sample(*args, **kwargs)
        finally:
            self._reuse_logits = False

    def apply_sampling_params(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        seq_lens_upper_bound_np,
        skip_top_k_top_p=False,
    ):
        if not np.any(self.needs_logits_processing[idx_mapping_np]):
            return logits
        if not self._reuse_logits or logits.dtype != torch.float32:
            logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
        else:
            self.reused_logits_calls += 1
        ctx = LogitsContext(
            expanded_idx_mapping=expanded_idx_mapping,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_local_pos=expanded_local_pos,
            input_ids=input_ids,
            pos=pos,
            seq_lens_upper_bound_np=seq_lens_upper_bound_np,
        )
        for processor in self.logits_processors:
            logits = processor.apply(logits, ctx)
        self.thinking_budget_state.apply(logits, ctx)
        self.sampling_states.apply_temperature(
            logits, expanded_idx_mapping, idx_mapping_np
        )
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)
        if skip_top_k_top_p:
            return logits
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )


class InplaceSamplerWorker:
    model_runner: "GPUModelRunner"

    def parallax_set_inplace_sampler(self, enabled):
        sampler = cast(Sampler, self.model_runner.sampler)
        if type(sampler) not in (Sampler, InplaceSampler):
            raise ValueError("Qualified V2 sampler instance required")
        sampler.__class__ = InplaceSampler if enabled else Sampler
        optimized = cast(InplaceSampler, sampler)
        optimized._reuse_logits = False
        optimized.reused_logits_calls = 0
        return {"enabled": enabled, "sampler": type(sampler).__name__}

    def parallax_inplace_sampler_report(self):
        sampler = cast(InplaceSampler, self.model_runner.sampler)
        return {"reused_logits_calls": sampler.reused_logits_calls}


class PipelineOptimizationWorker(DecodeSlotCacheWorker, InplaceSamplerWorker):
    """Combine independent cache bookkeeping and logits ownership optimizations."""
