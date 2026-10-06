# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lambda EDA using the GDN request-cache metadata contract."""

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.gdn.parallax_conv import causal_conv
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .parallax_eda_decode import decode
from .parallax_eda_pointwise import gates, norm_gate
from .parallax_eda_prefill import EDAInputs, prefill


class BatchedEDAProjection(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, erase, decay):
        return torch.bmm(torch.stack((erase, decay)), self.weight).unbind(0)


class EDALayer(GatedDeltaNetAttention):
    def __init__(self, c, vc, prefix):
        super().__init__(c, vc, prefix)
        self.h, self.d = c.gdn2_n_heads, c.gdn2_head_dim
        self.width = self.h * self.d
        self.round_before_silu = getattr(c, "lambda_conv_round", False)
        self.fused_pointwise = getattr(c, "lambda_fused_eda_pointwise", False)
        self.fused_decode = getattr(c, "lambda_fused_eda_decode", False)
        self.decode_grid_order = getattr(c, "eda_decode_grid_order", "sequence")
        self.decode_block_v = getattr(
            c,
            f"eda_decode_block_v_{c.gdn2_state_storage}",
            getattr(c, "eda_decode_block_v", 16),
        )
        self.decode_num_warps = getattr(
            c,
            f"eda_decode_num_warps_{c.gdn2_state_storage}",
            getattr(c, "eda_decode_num_warps", 4),
        )
        self.decode_small_block_v = getattr(c, "eda_decode_small_block_v", 16)
        self.decode_small_num_warps = getattr(
            c, "eda_decode_small_num_warps", self.decode_num_warps
        )
        self.decode_small_grid_order = getattr(
            c, "eda_decode_small_grid_order", self.decode_grid_order
        )
        if any(
            order not in ("sequence", "head")
            for order in (self.decode_grid_order, self.decode_small_grid_order)
        ):
            raise ValueError("EDA grid order must be sequence or head")
        self.joined_input = None
        self.joined_ef = None
        if self.decode_block_v not in (
            16,
            32,
            64,
            128,
        ) or self.decode_small_block_v not in (16, 32, 64, 128):
            raise ValueError("EDA decode block_v must be 16, 32, 64 or 128")
        if self.decode_num_warps not in (4, 8) or self.decode_small_num_warps not in (
            4,
            8,
        ):
            raise ValueError("EDA decode num_warps must be 4 or 8")
        self.state_storage_dtype = {"fp32": torch.float32, "fp16": torch.float16}[
            c.gdn2_state_storage
        ]
        self.norm = RMSNorm(c.d_model, eps=1e-6)
        self.q_proj = nn.Linear(c.d_model, self.width, bias=False)
        self.k_proj = nn.Linear(c.d_model, self.width, bias=False)
        self.v_proj = nn.Linear(c.d_model, self.width, bias=False)
        self.e_proj = nn.Sequential(
            nn.Linear(c.d_model, self.h * c.eda_erase_rank, bias=False),
            nn.Linear(self.h * c.eda_erase_rank, self.width, bias=False),
        )
        rank = c.eda_decay_rank or 16
        self.f_proj = nn.Sequential(
            nn.Linear(c.d_model, self.h * rank, bias=False),
            nn.Linear(self.h * rank, self.width, bias=False),
        )
        self.b_proj = nn.Linear(c.d_model, self.h, bias=True)
        self.c_proj = nn.Linear(c.d_model, self.h, bias=True)
        self.g_proj = nn.Sequential(
            nn.Linear(c.d_model, self.d, bias=False),
            nn.Linear(self.d, self.width, bias=True),
        )
        self.conv_weight = nn.Parameter(torch.empty(3 * self.width, 4))
        self.A_log = nn.Parameter(torch.zeros(self.h, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.width))
        self.o_norm = RMSNorm(self.d, eps=1e-6)
        self.o_proj = nn.Linear(self.width, c.d_model, bias=False)
        self.register_buffer(
            "_no_reset",
            torch.zeros(
                vc.scheduler_config.max_num_seqs,
                device=self.q_proj.weight.device,
                dtype=torch.bool,
            ),
            persistent=False,
        )
        vc.compilation_config.static_forward_context[prefix] = self

    def get_state_shape(self):
        return (3 * self.width, 3), (self.h, self.d, self.d)

    def get_state_dtype(self):
        return self.model_config.dtype, self.state_storage_dtype

    def forward(self, x, positions, normalized=False):
        x = x if normalized else self.norm(x)
        metadata = cast(
            dict[str, GDNAttentionMetadata] | None, get_forward_context().attn_metadata
        )
        if metadata is None or self.prefix not in metadata:
            return torch.zeros_like(x)
        m = metadata[self.prefix]
        if m.num_spec_decodes:
            raise NotImplementedError("EDA speculative state not qualified")
        n = m.num_actual_tokens
        starts, slots = m.non_spec_query_start_loc, m.non_spec_state_indices_tensor
        assert starts is not None and slots is not None
        slots = slots.to(torch.int32).contiguous()
        reset = (
            self._no_reset[: slots.numel()]
            if m.has_initial_state is None
            else ~m.has_initial_state
        )
        conv_state, state = self.kv_cache[:2]
        if self.joined_input is not None:
            projected = self.joined_input(x[:n])
            qkv, e0, f0, b, c, g0 = projected.split(
                [
                    3 * self.width,
                    self.e_proj[0].out_features,
                    self.f_proj[0].out_features,
                    self.h,
                    self.h,
                    self.d,
                ],
                dim=-1,
            )
            if self.joined_ef is not None:
                erase, decay = self.joined_ef(e0, f0)
            else:
                erase, decay = self.e_proj[1](e0), self.f_proj[1](f0)
            gate = self.g_proj[1](g0)
        else:
            qkv = torch.cat(
                [self.q_proj(x[:n]), self.k_proj(x[:n]), self.v_proj(x[:n])], -1
            )
            erase, decay = self.e_proj(x[:n]), self.f_proj(x[:n])
            b, c, gate = self.b_proj(x[:n]), self.c_proj(x[:n]), self.g_proj(x[:n])
        mixed = causal_conv(
            qkv,
            self.conv_weight,
            conv_state,
            starts,
            slots,
            reset,
            parallel=m.num_prefills > 0,
            round_before_silu=self.round_before_silu,
        )
        prepare, finish = gates, norm_gate
        if self.fused_pointwise:
            from .parallax_eda_fused import gates as prepare
            from .parallax_eda_fused import norm_gate as finish
        if self.fused_decode and m.num_prefills == 0:
            from .parallax_eda_gated import decode_gated

            y = decode_gated(
                mixed,
                erase,
                decay,
                b,
                c,
                self.A_log,
                self.dt_bias,
                state,
                starts,
                slots,
                reset,
                block_v=self.decode_block_v if n > 4 else self.decode_small_block_v,
                num_warps=self.decode_num_warps
                if n > 4
                else self.decode_small_num_warps,
                grid_order=self.decode_grid_order
                if n > 4
                else self.decode_small_grid_order,
            )
        else:
            args = prepare(
                mixed,
                erase,
                decay,
                b,
                c,
                self.A_log,
                self.dt_bias,
                self.h,
                self.d,
            )
            if m.num_prefills > 0:
                y = prefill(*cast(EDAInputs, args), state, starts, slots, reset)
            else:
                y = decode(
                    *args,
                    state,
                    starts,
                    slots,
                    reset,
                    block_v=self.decode_block_v if n > 4 else self.decode_small_block_v,
                    num_warps=self.decode_num_warps
                    if n > 4
                    else self.decode_small_num_warps,
                )
        y = finish(y, gate.reshape_as(y), self.o_norm.weight)
        out = self.o_proj(y.flatten(1))
        return out if x.shape[0] == n else F.pad(out, (0, 0, 0, x.shape[0] - n))
