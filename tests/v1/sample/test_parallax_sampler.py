# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parallax integration contract regressions."""

import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "dtype,reuse",
    [(torch.float32, True), (torch.float32, False), (torch.bfloat16, True)],
)
def test_sampler_preserves_processor_order_and_raw_logits(dtype, reuse):
    """Exercise the actual upstream processing contract without CUDA state setup."""
    from vllm.v1.worker.gpu.parallax_sampler import InplaceSampler
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    seen = []

    class Processor:
        def __init__(self, name, operation):
            self.name, self.operation = name, operation

        def apply(self, logits, ctx):
            seen.append((self.name, ctx.seq_lens_upper_bound_np.tolist()))
            self.operation(logits)
            return logits

    class Sampling:
        def apply_temperature(self, logits, *args):
            logits.div_(2)

        def apply_min_p(self, logits, *args):
            logits[logits < 0] = float("-inf")

        def apply_top_k_top_p(self, logits, *args):
            return logits

    def state(cls):
        obj = object.__new__(cls)
        obj.needs_logits_processing = np.array([True])
        obj.logits_processors = [
            Processor("bias", lambda x: x.add_(1)),
            Processor("custom", lambda x: x.mul_(3)),
        ]
        obj.thinking_budget_state = Processor(
            "forcing", lambda x: x.__setitem__((0, 1), -10)
        )
        obj.sampling_states = Sampling()
        obj._reuse_logits = reuse
        obj.reused_logits_calls = 0
        return obj

    raw = torch.tensor([[1, 2, 3]], dtype=dtype)
    original = raw.clone()
    args = (
        torch.tensor([0]),
        torch.tensor([0]),
        np.array([0]),
        torch.tensor([2]),
        torch.tensor([7]),
        torch.tensor([0]),
        np.array([3]),
    )
    expected = Sampler.apply_sampling_params(state(Sampler), original, *args)
    upstream_seen = seen[:]
    seen.clear()
    candidate = state(InplaceSampler)
    actual = candidate.apply_sampling_params(raw, *args)
    assert torch.equal(actual, expected)
    assert seen == upstream_seen == [("bias", [3]), ("custom", [3]), ("forcing", [3])]
    if reuse and dtype == torch.float32:
        assert actual.data_ptr() == raw.data_ptr()
        assert candidate.reused_logits_calls == 1
    else:
        assert torch.equal(raw, original)
        assert actual.data_ptr() != raw.data_ptr()


@pytest.mark.parametrize(
    "return_logprobs,mask", [(True, False), (False, True), (False, False)]
)
def test_sampler_positional_logprobs_controls_ownership(
    monkeypatch, return_logprobs, mask
):
    from vllm.v1.worker.gpu.parallax_sampler import InplaceSampler
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    candidate = object.__new__(InplaceSampler)
    candidate.return_sampling_mask = mask
    observed = []

    def sample(self, *args, **kwargs):
        observed.append(self._reuse_logits)
        return None

    monkeypatch.setattr(Sampler, "sample", sample)
    candidate.sample(*([None] * 8), return_logprobs)
    assert observed == [not return_logprobs and not mask]
    assert candidate._reuse_logits is False
