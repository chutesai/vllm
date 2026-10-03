# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fuse shared-expert pointwise operations, preserving BF16 round points."""

from vllm.triton_utils import tl, triton


@triton.jit
def _relu_square(X, TOTAL: tl.constexpr, B: tl.constexpr):
    pos = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.maximum(tl.load(X + pos, pos < TOTAL, other=0).to(tl.float32), 0)
    tl.store(X + pos, x * x, pos < TOTAL)


@triton.jit
def _scaled_add(X, SHARED, TOTAL: tl.constexpr, SCALE: tl.constexpr, B: tl.constexpr):
    pos = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + pos, pos < TOTAL, other=0).to(tl.float32)
    shared = tl.load(SHARED + pos, pos < TOTAL, other=0).to(tl.float32)
    scaled = (shared * SCALE).to(X.dtype.element_ty).to(tl.float32)
    tl.store(X + pos, x + scaled, pos < TOTAL)


def relu_square(x):
    assert x.is_cuda and x.is_contiguous()
    _relu_square[(triton.cdiv(x.numel(), 512),)](
        x, x.numel(), 512, enable_fp_fusion=False
    )
    return x


def scaled_add(x, shared, scale):
    assert x.shape == shared.shape and x.dtype == shared.dtype
    assert x.is_cuda and shared.is_cuda and x.is_contiguous() and shared.is_contiguous()
    _scaled_add[(triton.cdiv(x.numel(), 512),)](
        x, shared, x.numel(), scale, 512, enable_fp_fusion=False
    )
    return x
