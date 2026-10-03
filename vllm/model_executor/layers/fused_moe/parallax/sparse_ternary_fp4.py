# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental sparse FP4 experts with 32 output columns per warp.

Each warp reuses its activation fragments across two weight fragments and
reduces each 32-column quantization group without another warp.

Activation reconstruction is approximate; this must not be described as
lossless BF16 inference. The ternary codes and per-output weight scales remain
unchanged. Model quality must be qualified independently of component timing.
"""

import functools
import importlib

import torch

from vllm.model_executor.layers.fused_moe.parallax.sparse_ternary import (
    validate_pair_sparse,
)
from vllm.triton_utils import tl, triton

from ..moe_align_block_size import (
    moe_align_block_size,
)
from .ternary_decode import _combine


@functools.lru_cache(maxsize=1)
def extension():
    try:
        _parallax_C = importlib.import_module("vllm._parallax_C")
    except ImportError as exc:
        raise RuntimeError(
            "Sparse ternary FP4 requires this fork's compiled _parallax_C "
            "extension; see docs/models/parallax.md"
        ) from exc
    return _parallax_C


@triton.jit
def _combine_gated(
    X, G, Y, N: tl.constexpr, TOP: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr
):
    token = tl.program_id(0)
    # Match the original FP32 partial-buffer layout and reduction tree. Without
    # this constraint BF16 loads select a different layout and change rounding.
    ns = tl.max_contiguous(tl.program_id(1) * BN + tl.arange(0, BN), 4)
    ts = tl.arange(0, BT)
    x = tl.load(
        X + (token * TOP + ts[:, None]) * N + ns[None, :],
        (ts[:, None] < TOP) & (ns[None, :] < N),
        other=0,
    ).to(tl.float32)
    gates = tl.load(G + token * TOP + ts, ts < TOP, other=0).to(tl.float32)
    weighted = x * gates[:, None]
    tl.store(Y + token * N + ns, tl.sum(weighted, axis=0), ns < N)


class SparseTernaryFP4Experts:
    def __init__(self, up, down, alpha_up, alpha_down):
        validate_pair_sparse(up)
        validate_pair_sparse(down)
        self.ext = extension()
        self.up, self.um = self.ext.pack(up.contiguous())
        self.down, self.dm = self.ext.pack(down.contiguous())
        self.au, self.ad = (
            alpha_up.float().contiguous(),
            alpha_down.float().contiguous(),
        )
        self.experts = up.shape[0]

    def __call__(
        self,
        x,
        ids,
        gates,
        *,
        components=4,
        block_m=64,
        fused=True,
        ungated_reduce=False,
        combine_block_n=128,
        sorted_hidden=True,
    ):
        if (
            x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or components not in (1, 2, 3, 4)
        ):
            raise ValueError("Contiguous BF16 activations; one to four components")
        if sorted_hidden and not fused:
            raise ValueError("Sorted hidden layout requires fused up projection")
        ids = ids.to(torch.int32)
        top = ids.shape[1]
        sorted_ids, experts, count = moe_align_block_size(
            ids, block_m, self.experts, pad_sorted_ids=True
        )
        xp, xs, _ = self.ext.quantize(x, components, False)
        if fused:
            hp, hs = self.ext.run_up_quantized(
                xp,
                xs,
                self.up,
                self.um,
                self.au,
                sorted_ids,
                experts,
                count,
                ids.numel(),
                top,
                block_m,
                sorted_hidden,
            )
        else:
            h = self.ext.run(
                xp,
                xs,
                self.up,
                self.um,
                self.au,
                gates.reshape(-1),
                sorted_ids,
                experts,
                count,
                ids.numel(),
                top,
                True,
                block_m,
                False,
                False,
            )
            hp, hs, _ = self.ext.quantize(h, components, False)
        partial = self.ext.run(
            hp,
            hs,
            self.down,
            self.dm,
            self.ad,
            gates.reshape(-1),
            sorted_ids,
            experts,
            count,
            ids.numel(),
            top,
            False,
            block_m,
            ungated_reduce,
            sorted_hidden,
        )
        out = torch.empty_like(x)
        if ungated_reduce:
            if combine_block_n not in (64, 128, 256, 512):
                raise ValueError("Unsupported ordered reduction tile")
            _combine_gated[(x.shape[0], triton.cdiv(x.shape[1], combine_block_n))](
                partial,
                gates,
                out,
                x.shape[1],
                top,
                combine_block_n,
                triton.next_power_of_2(top),
                enable_fp_fusion=False,
            )
            return out
        _combine[(x.shape[0], triton.cdiv(x.shape[1], 128))](
            partial, out, x.shape[1], top, 128, triton.next_power_of_2(top)
        )
        return out
