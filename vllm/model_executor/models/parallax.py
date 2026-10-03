# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-GPU experimental Parallax model for the vLLM V1 scheduler.

All model families execute, including learned BAR and the separate LMSA selector.
Initial scope: BF16 trunk/KV, FP32 recurrent state, no prefix-cache sharing or
speculative decoding. Canonical loading uses model_loader/parallax.py.
Random initialization remains an explicit development option.
"""

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.parallax.sparse_ternary import (
    random_pair_sparse,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionMetadata,
)

from ..layers.attention import Attention
from ..layers.attention.attention import unified_kv_cache_update
from ..layers.attention.ops.parallax.paged_msa import paged_msa
from ..layers.fused_moe.parallax.fused_pointwise import (
    bar as fused_bar,
)
from ..layers.fused_moe.parallax.fused_pointwise import (
    gates as fused_gates,
)
from ..layers.fused_moe.parallax.fused_pointwise import norm_gate
from ..layers.fused_moe.parallax.ternary_decode import pack
from ..layers.layernorm import RMSNorm
from ..layers.logits_processor import LogitsProcessor
from ..layers.mamba.gdn.base import GatedDeltaNetAttention
from ..layers.mamba.gdn.parallax import decode_gated, recurrent
from ..layers.mamba.gdn.parallax_conv import causal_conv
from ..layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from ..layers.rotary_embedding import get_rope
from ..layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


def linear(din, dout, bias=False):
    return nn.Linear(din, dout, bias=bias)


def gdn_state_dtype(config):
    storage = getattr(config, "gdn2_state_storage", "fp32")
    if storage not in ("fp32", "fp16"):
        raise ValueError("GDN2 state storage must be fp32 or fp16")
    if storage == "fp16" and getattr(config, "rolling_gdn2_decode", False):
        raise ValueError("FP16 state is not qualified with rolling GDN2")
    return torch.float16 if storage == "fp16" else torch.float32


class BAR(nn.Module):
    def __init__(self, width, split=False, canonical=False):
        super().__init__()
        self.split = split
        self.canonical = canonical
        self.norm = RMSNorm(width, eps=1e-06)
        self.norm.eps = 1e-06
        self.proj = linear(width, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, sources, post_gamma=None):
        if self.canonical:
            from ..layers.attention.ops.parallax.block_residual import (
                fused_block_attn_res,
            )

            out = (
                sources[0]
                if len(sources) == 1
                else fused_block_attn_res(
                    self.proj.weight.squeeze(0),
                    self.norm.weight.to(sources[0].dtype),
                    sources,
                    1e-06,
                )
            )
            if post_gamma is not None:
                return F.rms_norm(
                    out, (out.shape[-1],), post_gamma.to(out.dtype), 1e-06
                )
            return out
        return fused_bar(
            sources,
            self.norm.weight,
            self.proj.weight,
            post_gamma,
            split=self.split and len(sources) >= 4 and (sources[0].shape[0] <= 16),
        )


class ExpertLayer(nn.Module):
    def __init__(self, c, layer_id, sparse_scratch=None):
        super().__init__()
        self.norm = RMSNorm(c.d_model, eps=1e-06)
        self.router = linear(c.d_model, c.n_routed_experts).float()
        self.latent_down = linear(c.d_model, c.moe_latent_dim)
        self.latent_up = linear(c.moe_latent_dim, c.d_model)
        self.top = c.num_experts_per_tok
        self.unique_expert_decode = getattr(c, "unique_expert_decode", False)
        self.single_sparse_fp4_components = getattr(
            c, "single_sparse_fp4_components", 0
        )
        if self.single_sparse_fp4_components not in (0, 1, 2):
            raise ValueError("Single sparse FP4 component setting must be 0, 1, or 2")
        self.single_sparse_fp4_kernel_name = getattr(
            c, "single_sparse_fp4_kernel", "sparse_ternary_fp4"
        )
        self.single_sparse_fp4_prefill = getattr(c, "single_sparse_fp4_prefill", False)
        self.single_sparse_fp4_ungated_reduce = getattr(
            c, "single_sparse_fp4_ungated_reduce", False
        )
        self.single_sparse_fp4_ungated_min_tokens = getattr(
            c, "single_sparse_fp4_ungated_min_tokens", 2048
        )
        if self.single_sparse_fp4_ungated_min_tokens < 256:
            raise ValueError("Ungated FP4 reduction requires at least 256 tokens")
        self.single_sparse_fp4_combine_block_n = getattr(
            c, "single_sparse_fp4_combine_block_n", 128
        )
        if self.single_sparse_fp4_combine_block_n not in (64, 128, 256, 512):
            raise ValueError("Unsupported ordered FP4 reduction tile")
        self.single_sparse_fp4_atomic_reduce = False
        self.single_sparse_fp4_decode_block_m = getattr(
            c, "single_sparse_fp4_decode_block_m", 32
        )
        self.single_sparse_fp4_decode_down_block_m = None
        if self.single_sparse_fp4_decode_block_m not in (16, 32, 64):
            raise ValueError("FP4 decode row tile must be 16, 32, or 64")
        self.native_sparse_bf16 = getattr(c, "native_sparse_bf16", False)
        self.expanded_sparse_prefill = getattr(c, "expanded_sparse_prefill", False)
        self.expanded_sparse_min_tokens = 2048
        self.sparse_halfpartial = False
        self.sparse_fused_gather = False
        self.expanded_sparse_block_k = getattr(c, "expanded_sparse_block_k", 128)
        self.expanded_sparse_tile = None
        if self.expanded_sparse_block_k not in (64, 128):
            raise ValueError("Expanded sparse reduction tile must be 64 or 128")
        if self.expanded_sparse_prefill and (not self.native_sparse_bf16):
            raise ValueError("Expanded sparse prefill requires native_sparse_bf16")
        self.pair_lut_decode = getattr(c, "pair_lut_decode", False)
        self.overlap_shared_experts = getattr(c, "overlap_shared_experts", False)
        self.overlap_shared_max_tokens = 512
        if not isinstance(512, int):
            raise ValueError("Positive shared-overlap token limit required")
        self.fused_shared_ops = getattr(c, "fused_shared_ops", False)
        self.fused_shared_bf16_gemm = getattr(c, "fused_shared_bf16_gemm", False)
        self.direct_sparse_decode = False
        self._shared_stream = None
        if self.overlap_shared_experts:
            from vllm.utils.torch_utils import aux_stream

            self._shared_stream = aux_stream()
            self._shared_input_ready = torch.cuda.Event()
            self._shared_output_ready = torch.cuda.Event()
        self.expert_block_n = getattr(c, "expert_block_n", 64)
        self.backend = c.expert_backend
        self.load_backend = "bf16" if self.backend == "packed_bf16" else self.backend
        self.fused_routing = getattr(c, "fused_routing", False)
        self.fast_router_sort = getattr(c, "fast_router_sort", False)
        self.projected_routing = False
        self.projected_routing_min_tokens = 0
        self.projected_routing_max_tokens = 128
        self.fused_expert_quantization = False
        self.expert_down_split = 1
        self.wide_expert_tiles = False
        self.wide_expert_min_tokens = 32
        self.fused_expert_min_tokens = 512
        self.scaling = c.routed_scaling_factor
        device = self.router.weight.device
        self.register_buffer(
            "qb_beta",
            torch.zeros(c.n_routed_experts, device=device, dtype=torch.float32),
        )
        self.shared_experts = nn.ModuleList()
        if not c.random_weights:
            if c.shared_expert_use_routed_latent or c.shared_latent_dim is not None:
                raise NotImplementedError(
                    "Shared latent configurations need qualification"
                )
            for _ in range(c.n_shared_experts):
                self.shared_experts.append(
                    nn.Sequential(
                        linear(
                            c.d_model,
                            c.shared_expert_intermediate_size
                            or c.moe_intermediate_size,
                        ),
                        nn.Identity(),
                        linear(
                            c.shared_expert_intermediate_size
                            or c.moe_intermediate_size,
                            c.d_model,
                        ),
                    )
                )
        self.shared_scale = c.shared_expert_output_scale
        for direction, n, k in (
            ("up", c.moe_intermediate_size, c.moe_latent_dim),
            ("down", c.moe_latent_dim, c.moe_intermediate_size),
        ):
            weights = []
            alphas = torch.full(
                (c.n_routed_experts,),
                1 / math.sqrt(k),
                device=device,
                dtype=torch.float32,
            )
            for e in range(c.n_routed_experts):
                if not c.random_weights:
                    if self.load_backend != "bf16":
                        raise NotImplementedError(
                            "Trained baseline requires BF16 experts"
                        )
                    weights.append(
                        torch.empty(
                            n, k, device=device, dtype=self.latent_down.weight.dtype
                        )
                    )
                    continue
                codes = random_pair_sparse(
                    n,
                    k,
                    device=device,
                    seed=123
                    + layer_id * 1000
                    + e
                    + (500 if direction == "down" else 0),
                )
                weights.append(
                    codes.to(self.latent_down.weight.dtype) * alphas[e]
                    if self.backend == "bf16"
                    else pack(codes)
                )
            self.register_buffer(direction, torch.stack(weights))
            self.register_buffer("alpha_" + direction, alphas)
        self.sparse = None

    def _shared_hidden(self, shared, x):
        if self.fused_shared_bf16_gemm and 256 <= x.shape[0] <= 512:
            from ..layers.fused_moe.parallax.dense_shared_bf16 import (
                linear,
            )

            tile = (
                (64, 64, 128, 4, 3)
                if x.shape[0] <= 512
                else (64, 64, 64, 4, 3)
                if x.shape[0] <= 4096
                else (128, 64, 64, 4, 3)
            )
            return linear(x, shared[0].weight, relu2=True, tile=tile)
        hidden = shared[0](x)
        if self.fused_shared_ops:
            from ..layers.fused_moe.parallax.shared_pointwise import (
                relu_square,
            )

            return relu_square(hidden)
        return hidden.relu().square()

    def forward(self, x, positions, normalized=False):
        x = x if normalized else self.norm(x)
        shared_outputs = None
        if self._shared_stream is not None and x.shape[0] <= 512:
            main_stream = torch.cuda.current_stream()
            self._shared_input_ready.record(main_stream)
            with torch.cuda.stream(self._shared_stream):
                self._shared_input_ready.wait(self._shared_stream)
                shared_outputs = []
                for shared in self.shared_experts:
                    hidden = self._shared_hidden(shared, x)
                    shared_result = shared[2](hidden)
                    if not self.fused_shared_ops:
                        shared_result = shared_result * (
                            self.shared_scale / len(self.shared_experts)
                        )
                    shared_outputs.append(shared_result)
                self._shared_output_ready.record(self._shared_stream)
        logits = self.router(x.float())
        if self.fused_routing:
            from ..layers.fused_moe.parallax.fused_pointwise import (
                route,
            )

            ids, gates = route(
                logits,
                self.qb_beta,
                self.top,
                self.scaling,
                fast_sort=self.fast_router_sort,
            )
        else:
            ids = (logits - self.qb_beta).topk(self.top, dim=-1).indices
            gates = logits.gather(1, ids).sigmoid()
            gates = (
                gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-06) * self.scaling
            ).contiguous()
        latent = self.latent_down(x)
        if self.backend == "bf16":
            from ..layers.fused_moe.parallax.packed_bf16 import (
                moe as dense_moe,
            )

            y = dense_moe(
                latent,
                self.up,
                self.down,
                self.alpha_up,
                self.alpha_down,
                ids,
                gates,
                block_m=32 if x.shape[0] >= 256 else 16,
                packed=False,
            )
        elif self.backend == "packed_bf16":
            from ..layers.fused_moe.parallax.packed_bf16 import (
                moe as packed_moe,
            )

            if self.single_sparse_fp4_components and (
                256 <= x.shape[0] <= 512
                or (self.single_sparse_fp4_prefill and x.shape[0] >= 256)
            ):
                fp4_options = {}
                fp4_options["sorted_hidden"] = x.shape[0] >= 2048
                if (
                    self.single_sparse_fp4_ungated_reduce
                    and x.shape[0] >= self.single_sparse_fp4_ungated_min_tokens
                ):
                    fp4_options["ungated_reduce"] = True
                    fp4_options["combine_block_n"] = (
                        self.single_sparse_fp4_combine_block_n
                    )
                y = self.single_sparse_fp4(
                    latent,
                    ids,
                    gates,
                    components=self.single_sparse_fp4_components,
                    block_m=64
                    if self.single_sparse_fp4_prefill and x.shape[0] >= 2048
                    else self.single_sparse_fp4_decode_block_m,
                    **fp4_options,
                )
            elif self.native_sparse_bf16 and x.shape[0] >= 256:
                from ..layers.fused_moe.parallax.packed_sparse_bf16 import (
                    moe as sparse_moe,
                )

                expanded = self.expanded_sparse_prefill and x.shape[0] >= 2048
                tile = None if expanded and x.shape[0] >= 8192 else None
                y = sparse_moe(
                    latent,
                    self.expanded_sparse_up if expanded else self.sparse_up,
                    self.expanded_sparse_down if expanded else self.sparse_down,
                    self.alpha_up,
                    self.alpha_down,
                    self.meta_up,
                    self.meta_down,
                    ids,
                    gates,
                    block_m=tile[0]
                    if tile
                    else 64
                    if expanded and x.shape[0] >= 2048
                    else 32,
                    block_n=tile[1] if tile else 128,
                    block_k=tile[2]
                    if tile
                    else self.expanded_sparse_block_k
                    if expanded and x.shape[0] >= 8192
                    else 128,
                    stages=tile[3] if tile else 2,
                    threads=tile[4] if tile else 128,
                    preexpanded=expanded,
                )
            elif self.pair_lut_decode and x.shape[0] <= 4:
                from ..layers.fused_moe.parallax.pair_lut_bf16 import (
                    moe as lut_moe,
                )

                y = lut_moe(
                    latent,
                    self.sparse_up,
                    self.sparse_down,
                    self.alpha_up,
                    self.alpha_down,
                    self.meta_up,
                    self.meta_down,
                    ids,
                    gates,
                    direct=False,
                )
            else:
                y = packed_moe(
                    latent,
                    self.up,
                    self.down,
                    self.alpha_up,
                    self.alpha_down,
                    ids,
                    gates,
                    block_m=32 if x.shape[0] >= 256 else 16,
                    block_n=self.expert_block_n if x.shape[0] >= 256 else 64,
                    unique_decode=self.unique_expert_decode,
                )
        else:
            raise ValueError(f"Unqualified expert backend: {self.backend}")
        out = self.latent_up(y)
        if shared_outputs is not None:
            self._shared_output_ready.wait(torch.cuda.current_stream())
            for shared_output in shared_outputs:
                if self.fused_shared_ops:
                    from ..layers.fused_moe.parallax.shared_pointwise import (
                        scaled_add,
                    )

                    out = scaled_add(
                        out, shared_output, self.shared_scale / len(self.shared_experts)
                    )
                else:
                    out = out + shared_output
        else:
            for shared in self.shared_experts:
                hidden = self._shared_hidden(shared, x)
                if self.fused_shared_ops:
                    from ..layers.fused_moe.parallax.shared_pointwise import (
                        scaled_add,
                    )

                    out = scaled_add(
                        out,
                        shared[2](hidden),
                        self.shared_scale / len(self.shared_experts),
                    )
                else:
                    out = out + shared[2](hidden) * (
                        self.shared_scale / len(self.shared_experts)
                    )
        return out


class GDN2Layer(GatedDeltaNetAttention):
    def __init__(self, c, vc, prefix):
        super().__init__(c, vc, prefix)
        self.h, self.d = (c.gdn2_n_heads, c.gdn2_head_dim)
        self.state_storage_dtype = gdn_state_dtype(c)
        self.round_state_prefill = False
        self.fused_decode_gates = getattr(c, "fused_decode_gates", False)
        self.parallel_prefill = getattr(c, "parallel_gdn2_prefill", False)
        self.parallel_prefill_max_seqs = 4
        self.prefill_state_tile = 32
        self.prefill_chunk = 32
        self.prefill_scan_schedule = None
        self.parallel_conv = getattr(c, "parallel_gdn2_conv", False)
        self.tuned_tiles = getattr(c, "tuned_gdn2_tiles", False)
        self.decode_block_v = getattr(c, "gdn2_decode_block_v", None)
        self.decode_kernel_options = {}
        if self.decode_block_v is not None:
            if self.decode_block_v not in (16, 32, 64, 128):
                raise ValueError("Unsupported GDN2 decode value tile")
            if self.state_storage_dtype not in (torch.float32, torch.float16):
                raise ValueError("GDN2 decode tuning requires FP32 or FP16 state")
            warps = getattr(c, "gdn2_decode_num_warps", 4)
            load_cache = getattr(c, "gdn2_decode_load_cache", "")
            store_cache = getattr(c, "gdn2_decode_store_cache", "")
            if warps not in (1, 2, 4, 8):
                raise ValueError("Unsupported GDN2 decode warp count")
            if load_cache not in ("", ".ca", ".cg") or store_cache not in (
                "",
                ".wb",
                ".cs",
                ".wt",
            ):
                raise ValueError("Unsupported GDN2 decode cache policy")
            self.decode_kernel_options = {
                "num_warps": warps,
                "state_load_cache": load_cache,
                "state_store_cache": store_cache,
            }
        self.decode_grid_order = getattr(c, "gdn2_decode_grid_order", None)
        if self.decode_grid_order is not None and self.decode_grid_order not in (
            0,
            1,
            2,
            3,
        ):
            raise ValueError("Unsupported GDN2 scheduling order")
        self.rolling_decode = False
        self.rolling_chunk = 8
        self.width = self.h * self.d
        self.packed_qkv_bw = None
        self.packed_fg = None
        self.joined_input = None
        self.joined_max_tokens = getattr(c, "gdn2_joined_max_tokens", None)
        if self.joined_max_tokens is not None and self.joined_max_tokens < 1:
            raise ValueError("Joined GDN2 token limit must be positive")
        self.tuned_joined_bf16_gemm = getattr(c, "tuned_joined_bf16_gemm", False)
        self.paired_fg_outputs = False
        self.norm = RMSNorm(c.d_model, eps=1e-06)
        self.qkv = linear(c.d_model, 3 * self.width)
        self.bw = linear(c.d_model, 2 * self.width)
        self.f_proj = nn.Sequential(
            linear(c.d_model, self.d), linear(self.d, self.width)
        )
        self.g_proj = nn.Sequential(
            linear(c.d_model, self.d), linear(self.d, self.width, bias=True)
        )
        self.conv_weight = nn.Parameter(torch.empty(3 * self.width, 4))
        nn.init.normal_(self.conv_weight, std=0.1)
        self.A_log = nn.Parameter(torch.zeros(self.h, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.full((self.width,), -4.0, dtype=torch.float32)
        )
        self.o_norm = RMSNorm(self.d, eps=1e-06)
        self.o_proj = linear(self.width, c.d_model)
        self.register_buffer(
            "_no_reset",
            torch.zeros(
                vc.scheduler_config.max_num_seqs,
                device=self.qkv.weight.device,
                dtype=torch.bool,
            ),
            persistent=False,
        )
        vc.compilation_config.static_forward_context[prefix] = self

    def get_state_shape(self):
        shapes = ((3 * self.width, 3), (self.h, self.d, self.d))
        return shapes

    def get_state_dtype(self):
        dtypes = (self.model_config.dtype, self.state_storage_dtype)
        return dtypes

    def forward(self, x, positions, normalized=False):
        x = x if normalized else self.norm(x)
        joined_projection = (
            self.joined_input
            if self.joined_max_tokens is None or x.shape[0] <= self.joined_max_tokens
            else None
        )
        if (
            self.tuned_joined_bf16_gemm
            and joined_projection is not None
            and (256 <= x.shape[0] <= 4096)
        ):
            from ..layers.fused_moe.parallax.dense_shared_bf16 import (
                linear,
            )

            joined = linear(
                x,
                joined_projection.weight,
                tile=(64, 64, 64, 4, 3) if x.shape[0] <= 512 else (128, 64, 64, 4, 3),
            )
        else:
            joined = joined_projection(x) if joined_projection is not None else None
        projected = (
            joined[:, : 5 * self.width]
            if joined is not None
            else self.packed_qkv_bw(x)
            if self.packed_qkv_bw is not None
            else None
        )
        mixed = self.qkv(x) if projected is None else projected[:, : 3 * self.width]
        metadata = cast(
            dict[str, GDNAttentionMetadata] | None, get_forward_context().attn_metadata
        )
        if metadata is None or self.prefix not in metadata:
            return self.o_proj(torch.zeros_like(x))
        m = metadata[self.prefix]
        if m.num_spec_decodes:
            raise NotImplementedError("GDN2 speculative state not qualified")
        n = m.num_actual_tokens
        starts = m.non_spec_query_start_loc
        state_indices = m.non_spec_state_indices_tensor
        assert starts is not None and state_indices is not None
        slots = state_indices.to(torch.int32).contiguous()
        reset = (
            self._no_reset[: slots.numel()]
            if m.has_initial_state is None
            else ~m.has_initial_state
        )
        conv_state, state = self.kv_cache[:2]
        mixed = causal_conv(
            mixed[:n],
            self.conv_weight,
            conv_state,
            starts,
            slots,
            reset,
            parallel=self.parallel_conv and m.num_prefills > 0,
        )
        bw = self.bw(x[:n]) if projected is None else projected[:n, 3 * self.width :]
        if joined is not None:
            fg = joined[:n, 5 * self.width :]
            f = self.f_proj[1](fg[:, : self.d])
            g = self.g_proj[1](fg[:, self.d :])
        elif self.packed_fg is None:
            f, g = (self.f_proj(x[:n]), self.g_proj(x[:n]))
        else:
            fg = self.packed_fg(x[:n])
            f = self.f_proj[1](fg[:, : self.d])
            g = self.g_proj[1](fg[:, self.d :])
        if self.fused_decode_gates and m.num_prefills == 0:
            state_decode = decode_gated
            if self.state_storage_dtype == torch.float16:
                from ..layers.mamba.gdn.parallax_fp16 import (
                    decode_gated as state_decode,
                )
            if self.decode_grid_order is not None:
                from ..layers.mamba.gdn.parallax_decode import (
                    decode_gated as state_decode,
                )
            grid_options = (
                {"grid_order": self.decode_grid_order}
                if self.decode_grid_order is not None
                else {}
            )
            y = state_decode(
                mixed,
                bw,
                f,
                self.A_log,
                self.dt_bias,
                state,
                starts,
                slots,
                reset,
                block_v=self.decode_block_v
                if self.decode_block_v is not None and slots.numel() >= 64
                else 32
                if self.tuned_tiles and slots.numel() >= 64
                else 16,
                **self.decode_kernel_options if slots.numel() >= 64 else {},
                **grid_options,
            )
        else:
            q, k, v, decay, b, w = fused_gates(
                mixed, bw, f, self.A_log, self.dt_bias, self.h, self.d
            )
            if self.parallel_prefill and n >= 128 and (slots.numel() <= 4):
                from ..layers.mamba.gdn.parallax_prefill import (
                    parallel_prefill,
                )

                if self.state_storage_dtype == torch.float16:
                    from ..layers.mamba.gdn.parallax_prefill_state16 import (
                        parallel_prefill,
                    )
                y = parallel_prefill(
                    q,
                    k,
                    v,
                    decay,
                    b,
                    w,
                    state,
                    starts,
                    slots,
                    reset,
                    state_tile=32,
                    chunk=32,
                )
            else:
                state_recurrent = recurrent
                state_options = {}
                if self.state_storage_dtype == torch.float16:
                    from ..layers.mamba.gdn.parallax_fp16 import (
                        recurrent as state_recurrent,
                    )

                    state_options["round_each_step"] = False
                y = state_recurrent(
                    q,
                    k,
                    v,
                    decay,
                    b,
                    w,
                    state,
                    starts,
                    slots,
                    reset,
                    block_v=32 if self.tuned_tiles and slots.numel() >= 8 else 16,
                    **state_options,
                )
        y = norm_gate(y, g.reshape_as(y), self.o_norm.weight)
        out = self.o_proj(y.flatten(1))
        return out if x.shape[0] == n else F.pad(out, (0, 0, 0, x.shape[0] - n))


class AttentionLayer(nn.Module):
    def __init__(self, c, vc, prefix, sparse):
        super().__init__()
        self.optimize_msa = False
        self.full_selection_fastpath = getattr(c, "msa_full_selection_fastpath", False)
        self.chunked_decode_index_scores = getattr(
            c, "msa_chunked_decode_index_scores", False
        )
        self.fused_msa_selection = getattr(c, "msa_fused_decode_selection", False)
        if self.fused_msa_selection and (not self.chunked_decode_index_scores):
            raise ValueError("Fused selection requires chunked index scoring")
        self.late_v_decode = False
        self.compact_cache = False
        if self.compact_cache and c.msa_kv_latent_dim != 256:
            raise ValueError("Compact LMSA requires latent=256")
        self.projected_key_cache = False
        self.packed_kv = None
        self.packed_index = None
        self.sparse = sparse
        self.tensorcore_index_scores = False
        self.msa_score_tile = 16
        self.msa_score_blocks = 1
        self.direct_sparse_attention = False
        self.bounded_sparse_attention = False
        self.tensorcore_index_min_context = 0
        self.tensorcore_sparse_attention = False
        self.fp32_sparse_probabilities = False
        self.tensorcore_sparse_prefill_only = False
        self.grouped_scalar_msa_prefill = getattr(
            c, "grouped_scalar_msa_prefill", False
        )
        self.grouped_scalar_maxnreg = None
        self.grouped_scalar_heads = 4
        self.dense_fp32_msa_prefill = getattr(c, "dense_fp32_msa_prefill", False)
        self.dense_probability_components = getattr(
            c, "dense_probability_components", 0
        )
        self.dense_probability_dtype = "bf16"
        self.dense_query_tile = getattr(c, "dense_query_tile", 32)
        self.dense_warps = getattr(c, "dense_warps", 4)
        self.tensorcore_sparse_min_tokens = 0
        self.hq, self.hk = (c.n_q_heads, c.n_kv_heads)
        self.d = c.msa_head_dim if sparse else 64
        self.norm = RMSNorm(c.d_model, eps=1e-06)
        self.q_proj = linear(c.d_model, self.hq * self.d)
        self.o_proj = linear(self.hq * self.d, c.d_model)
        self.rope = get_rope(
            self.d,
            max_position=c.max_seq_len,
            rope_parameters={"rope_type": "default", "rope_theta": c.rope_theta},
            is_neox_style=False,
        )
        backend = "triton"
        if backend not in ("triton", "flash_attn"):
            raise ValueError("dense_attention_backend must be triton or flash_attn")
        if vc.model_config.max_model_len < 0:
            backend = "triton"
        dense = not sparse or vc.model_config.max_model_len <= c.msa_sparse_topk * 128
        self.attn = Attention(
            self.hq,
            384 if self.projected_key_cache else self.d,
            self.d ** (-0.5),
            1 if self.compact_cache else self.hk,
            cache_config=vc.cache_config,
            prefix=prefix + ".attn",
            per_layer_sliding_window=None if sparse else c.sliding_window_size,
            attn_backend=FlashAttentionBackend
            if dense and backend == "flash_attn" and (not self.compact_cache)
            else TritonAttentionBackend,
        )
        if sparse:
            self.max_context = vc.model_config.max_model_len
            self.kv_down_proj = linear(c.d_model, c.msa_kv_latent_dim)
            self.kv_latent_norm = RMSNorm(c.msa_kv_latent_dim, eps=1e-06)
            self.k_proj = linear(c.msa_kv_latent_dim, self.hk * self.d)
            self.v_proj = linear(c.msa_kv_latent_dim, self.hk * self.d)
            rank = c.msa_index_rank or max(8, c.d_model // 4)
            self.index_q = nn.Sequential(
                linear(c.d_model, rank), linear(rank, self.hk * self.d)
            )
            self.index_k = nn.Sequential(linear(c.d_model, rank), linear(rank, self.d))
            self.index_cache = Attention(
                self.hk,
                64 if self.compact_cache else self.d,
                self.d ** (-0.5),
                1,
                cache_config=vc.cache_config,
                prefix=prefix + ".index_cache",
                attn_backend=TritonAttentionBackend,
            )
            self.topk = c.msa_sparse_topk
            self.hot_keys = None
            self.hot_tags = None
            if self.compact_cache and (not self.projected_key_cache):
                capacity = 8192
                if capacity < 0 or capacity > 32768:
                    raise ValueError("msa_hot_key_chunks must be within 0..32768")
                if capacity:
                    if self.hk != 4:
                        raise ValueError("Hot LMSA keys require four KV heads")
                    del self.hot_keys, self.hot_tags
                    self.register_buffer(
                        "hot_keys",
                        torch.empty(
                            capacity,
                            16,
                            4,
                            128,
                            device=self.q_proj.weight.device,
                            dtype=torch.bfloat16,
                        ),
                        persistent=False,
                    )
                    self.register_buffer(
                        "hot_tags",
                        torch.full(
                            (capacity,),
                            -1,
                            device=self.q_proj.weight.device,
                            dtype=torch.int64,
                        ),
                        persistent=False,
                    )
        else:
            self.k_proj = linear(c.d_model, self.hk * self.d)
            self.v_proj = linear(c.d_model, self.hk * self.d)

    def _forward_compact(self, x, latent, positions):
        q = self.q_proj(x)
        if self.packed_index is None:
            qi, ki = (self.index_q(x), self.index_k(x))
        else:
            packed = self.packed_index(x)
            rank = self.index_q[0].out_features
            qi = self.index_q[1](packed[:, :rank])
            ki = self.index_k[1](packed[:, rank:])
        projected_k = self.k_proj(latent) if self.projected_key_cache else None
        q, projected_k = self.rope(positions, q, projected_k)
        qi, ki = self.rope(positions, qi, ki)
        metadata = cast(
            dict[str, FlashAttentionMetadata | TritonAttentionMetadata] | None,
            get_forward_context().attn_metadata,
        )
        if metadata is None:
            return self.o_proj(torch.zeros_like(q))
        name, index_name = (self.attn.layer_name, self.index_cache.layer_name)
        m, im = (metadata[name], metadata[index_name])
        n = m.num_actual_tokens
        cache = self.attn.kv_cache.transpose(1, 2)
        index = self.index_cache.kv_cache.transpose(1, 2)
        from ..layers.attention.ops.parallax.compact_cache import (
            write_compact_cache,
        )

        slots = cast(dict[str, torch.Tensor], get_forward_context().slot_mapping)
        keys = None
        if self.projected_key_cache:
            from ..layers.attention.ops.parallax.key_latent_attention import (
                write_key_latent_cache,
            )

            cache = cache.view(cache.shape[0], cache.shape[1], 1, 768)
            write_key_latent_cache(
                projected_k, latent, ki, slots[name], slots[index_name], cache, index
            )
            keys = cache[..., :512].view(cache.shape[0], cache.shape[1], 4, 128)
            cache = cache[..., 512:]
        else:
            write_compact_cache(
                latent,
                ki,
                slots[name],
                slots[index_name],
                cache,
                index,
                hot_tags=self.hot_tags,
            )
        y, _ = paged_msa(
            q[:n].view(n, self.hq, 128).contiguous(),
            qi[:n].view(n, self.hk, 128).contiguous(),
            cache,
            cache,
            index,
            m.block_table,
            im.block_table,
            m.query_start_loc,
            m.seq_lens,
            m.max_seq_len if m.max_query_len > 1 else self.max_context,
            self.topk,
            tensorcore_scores=False,
            max_query_len=m.max_query_len,
            score_tile=16 if m.max_query_len >= 128 else 16,
            score_blocks=1,
            latent_projection=(
                self.k_proj.weight,
                self.v_proj.weight,
                self.rope.cos_sin_cache,
            ),
            latent_keys=keys,
            latent_hot=(self.hot_keys, self.hot_tags)
            if self.hot_keys is not None
            else None,
        )
        return self.o_proj(y.reshape(n, -1))

    def forward(self, x, positions, normalized=False):
        x = x if normalized else self.norm(x)
        src = self.kv_latent_norm(self.kv_down_proj(x)) if self.sparse else x
        if self.compact_cache:
            return self._forward_compact(x, src, positions)
        if self.packed_kv is None:
            k, v = (self.k_proj(src), self.v_proj(src))
        else:
            k, v = self.packed_kv(src).split(self.hk * self.d, dim=-1)
        q, k = self.rope(positions, self.q_proj(x), k)
        if not self.sparse:
            return self.o_proj(self.attn(q, k, v))
        if self.max_context <= self.topk * 128:
            return self.o_proj(self.attn(q, k, v))
        if self.full_selection_fastpath:
            metadata = cast(
                dict[str, FlashAttentionMetadata | TritonAttentionMetadata] | None,
                get_forward_context().attn_metadata,
            )
            m = None if metadata is None else metadata[self.attn.layer_name]
            if m is not None and m.max_seq_len <= self.topk * 128:
                if self.packed_index is None:
                    ki = self.index_k(x)
                else:
                    packed = self.packed_index(x)
                    rank = self.index_q[0].out_features
                    ki = self.index_k[1](packed[:, rank:])
                ki, _ = self.rope(positions, ki, None)
                ik = ki.view(-1, 1, self.d)
                unified_kv_cache_update(ik, ik, self.index_cache.layer_name)
                return self.o_proj(self.attn(q, k, v))
        if self.packed_index is None:
            qi, ki = (self.index_q(x), self.index_k(x))
        else:
            packed = self.packed_index(x)
            rank = self.index_q[0].out_features
            qi = self.index_q[1](packed[:, :rank])
            ki = self.index_k[1](packed[:, rank:])
        qi, ki = self.rope(positions, qi, ki)
        metadata = cast(
            dict[str, FlashAttentionMetadata | TritonAttentionMetadata] | None,
            get_forward_context().attn_metadata,
        )
        if metadata is None:
            return self.o_proj(torch.zeros_like(q))
        name, index_name = (self.attn.layer_name, self.index_cache.layer_name)
        m, im = (metadata[name], metadata[index_name])
        n = m.num_actual_tokens
        unified_kv_cache_update(
            k.view(-1, self.hk, self.d), v.view(-1, self.hk, self.d), name
        )
        ik = ki.view(-1, 1, self.d)
        unified_kv_cache_update(ik, ik, index_name)
        kc, vc = self.attn.kv_cache.transpose(1, 2).split(self.d, dim=-1)
        ikc = self.index_cache.kv_cache.transpose(1, 2)[..., : self.d]
        msa_impl = paged_msa
        msa_options = {}
        if self.chunked_decode_index_scores and m.max_query_len == 1:
            from ..layers.attention.ops.parallax.paged_msa_chunked_scores import (
                paged_msa as msa_impl,
            )

            msa_options["late_v_decode"] = False
            msa_options["fused_selection"] = self.fused_msa_selection
        y, _ = msa_impl(
            q[:n].view(n, self.hq, self.d).contiguous(),
            qi[:n].view(n, self.hk, self.d).contiguous(),
            kc,
            vc,
            ikc,
            m.block_table,
            im.block_table,
            m.query_start_loc,
            m.seq_lens,
            self.max_context,
            self.topk,
            tensorcore_scores=False,
            max_query_len=m.max_query_len,
            score_tile=16 if m.max_query_len >= 128 else 16,
            score_blocks=1 if m.max_query_len >= 128 else 1,
            direct_attend=False,
            bounded_attend=False,
            split_decode=False,
            tensorcore_attend=False,
            fp32_probabilities=False,
            grouped_scalar=self.grouped_scalar_msa_prefill and m.max_query_len > 1,
            grouped_scalar_maxnreg=None,
            grouped_scalar_heads=4,
            dense_probability_components=self.dense_probability_components,
            dense_probability_dtype="bf16",
            dense_query_tile=self.dense_query_tile,
            dense_warps=self.dense_warps,
            **msa_options,
            dense_prefix_context=m.max_seq_len
            if self.dense_fp32_msa_prefill
            and m.max_query_len > 1
            and (m.max_seq_len <= self.topk * 128)
            else None,
        )
        y = F.pad(y.flatten(1), (0, 0, 0, x.shape[0] - n))
        return self.o_proj(y)


class ParallaxForCausalLM(nn.Module):
    has_inner_state = True
    is_hybrid = True

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        c = vllm_config.model_config.hf_config
        self.config = c
        self.cache_bar_parameters = False
        self._bar_static_cache = {}
        if getattr(c, "bf16_full_accum", False):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.fused_input_norm = False
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError("This qualification targets one GPU")
        if vllm_config.cache_config.enable_prefix_caching:
            raise NotImplementedError("GDN2 prefix-cache reuse not qualified")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError("Speculative GDN2 rollback not qualified")
        if (
            not c.random_weights
            and getattr(c, "checkpoint_format", None) != "kappa_bf16"
        ):
            raise NotImplementedError("A verified kappa_bf16 checkpoint is required")
        if not c.random_weights and getattr(c, "dense_precision", "bf16") != "bf16":
            raise NotImplementedError("Qualify the BF16 checkpoint before quantizing")
        self.input_scale = (
            (math.sqrt(c.d_model) if c.tied_input_scale is None else c.tied_input_scale)
            if c.tie_word_embeddings
            else 1.0
        )
        self.register_buffer("logit_scale", torch.ones(1, dtype=torch.float32))
        self.embed = VocabParallelEmbedding(c.vocab_size, c.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.layers = nn.ModuleList()
        self.block_attn_res = nn.ModuleList()
        sparse_scratch = None
        for i, kind in enumerate(c.hybrid_override_pattern):
            name = f"layers.{i}"
            if kind == "E":
                layer = ExpertLayer(c, i, sparse_scratch)
            elif kind == "R":
                layer = GDN2Layer(c, vllm_config, name)
            elif kind in "W*":
                layer = AttentionLayer(c, vllm_config, name, kind == "*")
            else:
                raise ValueError(f"Unsupported layer {kind}")
            self.layers.append(layer)
            self.block_attn_res.append(
                BAR(c.d_model, False, canonical=not c.random_weights)
            )
        self.block_attn_res_head = BAR(c.d_model, False, canonical=not c.random_weights)
        self.block_size = len(self.layers) // c.block_attn_res_n_blocks
        self.final_norm = RMSNorm(c.d_model, eps=1e-06)
        self.lm_head = ParallelLMHead(c.vocab_size, c.d_model)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        self.logits_processor = LogitsProcessor(c.vocab_size)
        self.fp8_head = None
        if getattr(c, "dense_precision", "bf16") != "bf16":
            raise ValueError("Trained Parallax requires BF16 trunk projections")

    def embed_input_ids(self, input_ids):
        return self.embed(input_ids) * self.input_scale

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        x = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        blocks = [x]
        if not self.config.random_weights:
            from ..layers.attention.block_attention_residual import (
                BlockAttnRes,
            )

            for start in range(0, len(self.layers), self.block_size):
                end = min(start + self.block_size, len(self.layers))
                modules = self.block_attn_res[start:end]
                iw, im, denom = BlockAttnRes.batched_inter_stats(modules, blocks)
                partial = None
                for j, i in enumerate(range(start, end)):
                    module = modules[j]
                    if partial is None:
                        h = BlockAttnRes.stats_to_output(module, iw[j], denom[j])
                    else:
                        h = BlockAttnRes.merge_inter_stats_with_partial(
                            module, iw[j], im[j], denom[j], partial
                        )
                    branch = self.layers[i](h, positions)
                    if getattr(self.config, "fused_residual_accumulation", False):
                        from ..layers.fused_moe.parallax.residual import (
                            update,
                        )

                        partial = update(h, branch, partial)
                    else:
                        delta = h + branch - h
                        partial = delta if partial is None else partial + delta
                blocks.append(partial)
            return self.final_norm(self.block_attn_res_head(blocks))
        for start in range(0, len(self.layers), self.block_size):
            partial = None
            for i in range(start, min(start + self.block_size, len(self.layers))):
                sources = blocks if partial is None else blocks + [partial]
                gamma = None
                h = self.block_attn_res[i](sources, gamma)
                delta = self.layers[i](h, positions, normalized=False)
                partial = delta if partial is None else partial + delta
            blocks.append(partial)
        return self.final_norm(self.block_attn_res_head(blocks))

    def compute_logits(self, hidden_states):
        if self.fp8_head is not None:
            return self.fp8_head(hidden_states)[..., : self.config.vocab_size]
        return self.logits_processor(
            self.lm_head, hidden_states * self.logit_scale[0].to(hidden_states.dtype)
        )

    def load_weights(self, weights):
        from vllm.model_executor.model_loader.parallax import load_kappa_weights

        self._bar_static_cache.clear()
        return load_kappa_weights(self, weights)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vc):
        dtypes: tuple[torch.dtype, ...] = (
            vc.model_config.dtype,
            gdn_state_dtype(vc.model_config.hf_config),
        )
        if getattr(vc.model_config.hf_config, "rolling_gdn2_decode", False):
            dtypes += (torch.float32,)
        return dtypes

    @classmethod
    def get_mamba_state_shape_from_config(cls, vc):
        c = vc.model_config.hf_config
        shapes = (
            (3 * c.gdn2_n_heads * c.gdn2_head_dim, 3),
            (c.gdn2_n_heads, c.gdn2_head_dim, c.gdn2_head_dim),
        )
        return shapes

    def get_mamba_state_copy_funcs(self, mamba_types):
        funcs = self.get_mamba_state_copy_func()
        if getattr(self.config, "rolling_gdn2_decode", False):
            funcs += (funcs[1],)
        return {kind: funcs for kind in mamba_types}

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
