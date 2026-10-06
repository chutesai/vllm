# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lambda BF16 expert arithmetic must preserve mesh's ordered rounding."""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.parallax.packed_bf16 import moe


def test_bf16_experts_round_products_and_accumulate_in_expert_order():
    torch.manual_seed(12)
    x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(4, 256, 128, device="cuda", dtype=x.dtype) * 0.03
    down = torch.randn(4, 128, 256, device="cuda", dtype=x.dtype) * 0.03
    ids = torch.tensor(
        [[3, 0, 2], [1, 0, 3], [2, 3, 1], [3, 1, 0], [0, 1, 2], [3, 2, 1], [2, 1, 0]],
        device="cuda",
    )
    gates = torch.rand(7, 3, device="cuda").bfloat16()
    gates = (gates / gates.sum(-1, keepdim=True)).float()
    ref = torch.zeros_like(x)
    for expert in range(4):
        tokens, routes = (ids == expert).nonzero(as_tuple=True)
        out = F.linear(F.linear(x[tokens], up[expert]).relu().square(), down[expert])
        ref[tokens] += out * gates[tokens, routes, None]
    scale = torch.ones(4, device="cuda")
    actual = moe(
        x, up, down, scale, scale, ids, gates, packed=False, mesh_bf16_reduce=True
    )
    torch.testing.assert_close(actual, ref, atol=0.001, rtol=0.01)
