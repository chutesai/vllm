# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native configuration for the Parallax hybrid model."""

import json
from pathlib import Path

from transformers import PreTrainedConfig


class ParallaxConfig(PreTrainedConfig):
    model_type = "parallax"

    def __init__(self, **kwargs):
        defaults = json.loads(
            Path(__file__).with_name("parallax_defaults.json").read_text()
        )
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.hidden_size = self.d_model
        self.num_hidden_layers = len(self.hybrid_override_pattern)
        self.num_attention_heads = self.n_q_heads
        self.num_key_value_heads = self.n_kv_heads
        # vLLM's common cache planning uses the largest full-attention head.
        self.head_dim = self.msa_head_dim
        self.max_position_embeddings = self.max_seq_len
        self.hidden_act = "silu"
        self.rms_norm_eps = 1e-6
        self.architectures = ["ParallaxForCausalLM"]
        self.layers_block_type = [
            "mamba" if x == "R" else "mlp" if x == "E" else "attention"
            for x in self.hybrid_override_pattern
        ]
        self.expert_backend = defaults.get("expert_backend", "bf16")
        self.random_weights = defaults.get("random_weights", False)

        if self.expert_backend not in ("bf16", "packed_bf16"):
            raise ValueError("Parallax supports bf16 and packed_bf16 expert backends")
        kernel = getattr(self, "single_sparse_fp4_kernel", "sparse_ternary_fp4")
        if kernel not in ("v28", "sparse_ternary_fp4"):
            raise ValueError("Retired sparse-FP4 kernel; use sparse_ternary_fp4")
        self.single_sparse_fp4_kernel = "sparse_ternary_fp4"
        retired = {
            "single_sparse_fp4_atomic_reduce": False,
            "single_sparse_fp4_decode_down_block_m": None,
            "expanded_sparse_min_tokens": 2048,
            "sparse_halfpartial": False,
            "sparse_fused_gather": False,
            "expanded_sparse_tile": None,
            "overlap_shared_max_tokens": 512,
            "direct_sparse_decode": False,
            "projected_routing": False,
            "projected_routing_min_tokens": 0,
            "projected_routing_max_tokens": 128,
            "fused_expert_quantization": False,
            "expert_down_split": 1,
            "wide_expert_tiles": False,
            "wide_expert_min_tokens": 32,
            "fused_expert_min_tokens": 512,
            "gdn2_state_round_prefill": False,
            "gdn2_parallel_prefill_max_seqs": 4,
            "gdn2_prefill_state_tile": 32,
            "gdn2_prefill_chunk": 32,
            "gdn2_prefill_scan_schedule": None,
            "rolling_gdn2_decode": False,
            "rolling_gdn2_chunk": 8,
            "paired_gdn2_outputs": False,
            "optimized_msa": False,
            "msa_late_v_decode": False,
            "tensorcore_index_scores": False,
            "msa_score_tile": 16,
            "msa_score_blocks": 1,
            "direct_sparse_attention": False,
            "bounded_sparse_attention": False,
            "tensorcore_index_min_context": 0,
            "tensorcore_sparse_attention": False,
            "fp32_sparse_probabilities": False,
            "tensorcore_sparse_prefill_only": False,
            "grouped_scalar_maxnreg": None,
            "grouped_scalar_heads": 4,
            "dense_probability_dtype": "bf16",
            "tensorcore_sparse_min_tokens": 0,
            "dense_attention_backend": "triton",
            "cache_bar_parameters": False,
            "fused_input_norm": False,
            "packed_gdn2_projections": False,
            "packed_attention_projections": False,
            "bf16_decode_projections": False,
            "small_row_fp8": False,
            "component_timing": False,
            "compact_msa_cache": False,
            "msa_cache_projected_keys": False,
            "dense_attention_min_context": 0,
            "split_bar": False,
            "msa_hot_key_chunks": 8192,
            "tuned_fp8_pipeline": False,
        }
        for key, default in retired.items():
            if key in kwargs and kwargs[key] != default:
                raise ValueError(f"Retired Parallax experimental option: {key}")
