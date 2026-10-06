# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager FlashQLA packed EDA with request-cache gather and final scatter."""

import torch

from .parallax_eda_decode import recurrent

EDAInputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    e: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    state: torch.Tensor,
    starts: torch.Tensor,
    slots: torch.Tensor,
    reset: torch.Tensor,
) -> torch.Tensor:
    from flash_qla import chunk_eda

    # Empty sequences and padding have no FlashQLA state ownership contract.
    if q.shape[0] < 32 or bool(((slots < 0) | (starts[1:] == starts[:-1])).any()):
        return recurrent(q, k, v, e, g, beta, gamma, state, starts, slots, reset)
    initial = state.index_select(0, slots.long()).float()
    initial = torch.where(reset[:, None, None, None], 0.0, initial).contiguous()
    output, final = chunk_eda(
        *[x.unsqueeze(0) for x in (q, k, v, e, g, beta, gamma)],
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=starts.to(torch.int32),
        scale=q.shape[-1] ** -0.5,
        backend="tilelang",
    )
    state.index_copy_(0, slots.long(), final.to(state.dtype))
    return output.squeeze(0)
