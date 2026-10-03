# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fuse BAR residual bookkeeping while retaining every BF16 rounding point."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _update(
    H, BRANCH, PARTIAL, OUT, N: tl.constexpr, HAS_PARTIAL: tl.constexpr, B: tl.constexpr
):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    h = tl.load(H + offsets, offsets < N, other=0).to(tl.float32)
    branch = tl.load(BRANCH + offsets, offsets < N, other=0).to(tl.float32)
    summed = (h + branch).to(tl.bfloat16).to(tl.float32)
    delta = (summed - h).to(tl.bfloat16).to(tl.float32)
    if HAS_PARTIAL:
        delta += tl.load(PARTIAL + offsets, offsets < N, other=0).to(tl.float32)
    tl.store(OUT + offsets, delta, offsets < N)


def update(h, branch, partial=None):
    operands = (h, branch) if partial is None else (h, branch, partial)
    if any(
        x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape != h.shape
        or x.device != h.device
        for x in operands
    ):
        raise ValueError("Contiguous CUDA BF16 residual operands required")
    out = torch.empty_like(h)
    _update[(triton.cdiv(h.numel(), 256),)](
        h,
        branch,
        partial,
        out,
        h.numel(),
        partial is not None,
        256,
        enable_fp_fusion=False,
    )
    return out
