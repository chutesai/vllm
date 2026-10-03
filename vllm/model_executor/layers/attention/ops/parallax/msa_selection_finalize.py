# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact integer selection fix-up after PyTorch's unchanged score top-k."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _finalize(
    VALS, IDS, POS, OUT, GROUPS: tl.constexpr, K: tl.constexpr, BK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BK)
    values = tl.load(VALS + row * K + cols, cols < K, other=-float("inf"))
    indices = tl.load(IDS + row * K + cols, cols < K, other=-1)
    finite = (values > -float("inf")) & (values < float("inf"))
    indices = tl.where(finite, indices, -1)
    position = tl.load(POS + row // GROUPS)
    # Valid positions are nonnegative, graph padding uses -1. Avoid relying
    # on a backend's signed integer division convention for the padded row.
    local = tl.where(position < 0, -1, position // 128)
    present = tl.max(((indices == local) & (cols < K)).to(tl.int32), axis=0)
    indices = tl.where((cols == K - 1) & (present == 0), local, indices)
    tl.store(OUT + row * K + cols, indices.to(tl.int32), cols < K)


def finalize_selection(values, indices, positions):
    if (
        values.shape != indices.shape
        or values.ndim != 3
        or positions.shape != (values.shape[0],)
    ):
        raise ValueError(
            "Matching [tokens, groups, top-k] values/indices and positions required"
        )
    if (
        indices.dtype != torch.int64
        or values.dtype != torch.float32
        or positions.dtype != torch.int32
    ):
        raise ValueError(
            "FP32 scores, int64 top-k indices and int32 positions required"
        )
    if not all(
        t.is_cuda and t.is_contiguous() and t.device == values.device
        for t in (values, indices, positions)
    ):
        raise ValueError("Contiguous tensors on the same CUDA device required")
    tokens, groups, k = values.shape
    if not 1 <= k <= 128:
        raise ValueError("Supported top-k range is 1..128")
    out = torch.empty_like(indices, dtype=torch.int32)
    _finalize[(tokens * groups,)](
        values,
        indices,
        positions,
        out,
        groups,
        k,
        triton.next_power_of_2(k),
        num_warps=1,
    )
    return out
