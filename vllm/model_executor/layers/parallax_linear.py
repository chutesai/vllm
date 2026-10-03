# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Join BF16 projections sharing an input without changing row arithmetic."""

import torch
from torch import nn


def join_linears(layers):
    """Pack projections with one input, preserving row order."""
    if not all(isinstance(layer, nn.Linear) for layer in layers):
        raise ValueError("Joined projections must use one precision")
    if len({layer.weight.shape[1] for layer in layers}) != 1:
        raise ValueError("Joined projections must share the input width")
    weight = torch.cat([layer.weight for layer in layers])
    result = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
    result.weight = nn.Parameter(weight, requires_grad=False)
    if any(layer.bias is not None for layer in layers):
        result.bias = nn.Parameter(
            torch.cat(
                [
                    layer.bias
                    if layer.bias is not None
                    else torch.zeros(
                        layer.weight.shape[0], device=weight.device, dtype=weight.dtype
                    )
                    for layer in layers
                ]
            ),
            requires_grad=False,
        )
    return result
