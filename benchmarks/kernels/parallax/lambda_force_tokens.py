# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ground-truth continuation processor for cached-decode NLL measurements."""

import torch

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor


class ForceTokens(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        self.states = req_states
        self.targets = torch.zeros(
            (req_states.max_num_reqs, vllm_config.model_config.max_model_len),
            dtype=torch.long,
            device=req_states.device,
        )
        self.active = torch.zeros(
            req_states.max_num_reqs, dtype=torch.bool, device=req_states.device
        )

    def add_request(self, req_idx, sampling_params):
        tokens = (sampling_params.extra_args or {}).get("forced_tokens")
        self.active[req_idx] = tokens is not None
        if tokens is not None:
            self.targets[req_idx, : len(tokens)] = torch.tensor(
                tokens, device=self.targets.device
            )
        return tokens is not None

    def apply(self, logits, ctx):
        slots = ctx.expanded_idx_mapping.long()
        offsets = ctx.pos + 1 - self.states.prompt_len.gpu[slots]
        tokens = self.targets[slots, offsets.long()]
        active = self.active[slots]
        rows = torch.arange(logits.shape[0], device=logits.device)
        logits.masked_fill_(active[:, None], float("-inf"))
        logits[rows, tokens] = torch.where(active, 0.0, logits[rows, tokens])
        return logits
