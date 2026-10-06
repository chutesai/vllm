# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Create a validated execution-profile view of a converted Kappa checkpoint."""

import argparse
import json
from pathlib import Path

GEOMETRY = (
    "d_model",
    "hybrid_override_pattern",
    "moe_latent_dim",
    "moe_intermediate_size",
    "n_routed_experts",
    "num_experts_per_tok",
    "n_q_heads",
    "n_kv_heads",
    "msa_head_dim",
    "gdn2_head_dim",
    "gdn2_n_heads",
    "vocab_size",
    "tie_word_embeddings",
)
# Architecture, checkpoint policy and learned scales always come from the source.
OPTIONS = (
    "expert_backend",
    "dense_precision",
    "bf16_full_accum",
    "fused_routing",
    "fast_router_sort",
    "fused_decode_gates",
    "parallel_gdn2_conv",
    "parallel_gdn2_prefill",
    "tuned_gdn2_tiles",
    "head_dtype",
    "fused_residual_accumulation",
    "unique_expert_decode",
    "expert_block_n",
    "native_sparse_bf16",
    "pair_lut_decode",
    "overlap_shared_experts",
    "fused_shared_ops",
    "expanded_sparse_prefill",
    "expanded_sparse_block_k",
    "dense_fp32_msa_prefill",
    "grouped_scalar_msa_prefill",
    "msa_full_selection_fastpath",
    "joined_bf16_gdn2",
    "dense_probability_components",
    "dense_query_tile",
    "dense_warps",
    "single_sparse_fp4_components",
    "single_sparse_fp4_kernel",
    "single_sparse_fp4_prefill",
    "single_sparse_fp4_ungated_reduce",
    "gdn2_joined_max_tokens",
    "tuned_joined_bf16_gemm",
    "fused_shared_bf16_gemm",
    "single_sparse_fp4_decode_block_m",
    "gdn2_decode_block_v",
    "gdn2_decode_num_warps",
    "gdn2_decode_load_cache",
    "gdn2_decode_store_cache",
    "single_sparse_fp4_ungated_min_tokens",
    "single_sparse_fp4_combine_block_n",
    "gdn2_decode_grid_order",
    "msa_chunked_decode_index_scores",
    "msa_fused_decode_selection",
)


def prepare(source, output, eda_optimization="auto"):
    source, output = Path(source).resolve(), Path(output).resolve()
    config = json.loads((source / "config.json").read_text())
    profile = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "vllm/transformers_utils/configs/parallax_sm120.json"
        ).read_text()
    )
    is_eda = config.get("checkpoint_format") == "lambda_bf16"
    if eda_optimization == "auto":
        eda_optimization = "stable" if is_eda else "none"
    if eda_optimization not in (
        "none",
        "pointwise",
        "tiles",
        "combined",
        "joined",
        "gated",
        "tuned",
        "stable",
    ):
        raise ValueError("Unknown EDA optimization")
    if eda_optimization != "none" and not is_eda:
        raise ValueError("EDA optimizations require a Lambda checkpoint")
    if config.get("checkpoint_format") not in (
        "kappa_bf16",
        "lambda_bf16",
    ) or config.get("random_weights", False):
        raise ValueError("A converted, trained Parallax checkpoint is required")
    if is_eda:
        profile.update(
            hybrid_override_pattern="WERERERE*EREWERERE*EWEREREWE*EREREWERE*EREWERERE*EWERERERE*EREWE",
            tie_word_embeddings=True,
        )
        qualified = {
            "recurrent_backend": "eda",
            "eda_erase_rank": 16,
            "eda_gate_lower": -5.0,
            "eda_onorm_eps": 1e-6,
            "eda_scale_mode": "dk^-0.5",
            "n_shared_experts": 1,
            "final_norm_unit_gain": True,
            "moe_router_fp32": True,
            "sliding_window_size": 2048,
            "logit_scale_max": 3.0,
        }
        for key, expected in qualified.items():
            if config.get(key) != expected:
                raise ValueError(f"Unqualified Lambda policy: {key}")
        if config.get("eda_decay_rank") not in (None, 16):
            raise ValueError("Unqualified Lambda decay rank")
    for key in GEOMETRY:
        if config.get(key) != profile[key]:
            raise ValueError(f"Unqualified model geometry: {key}")
    if not (source / "model.safetensors.index.json").is_file():
        raise ValueError("Missing converted safetensors index")
    if source == output or output.exists():
        raise ValueError("Output must be a new directory, separate from the source")
    config.update({key: profile[key] for key in OPTIONS})
    if is_eda:
        config.update(
            joined_bf16_gdn2=False,
            inference_attention_mode="dense",
            pair_lut_decode=False,
            head_dtype="bfloat16",
            lambda_fused_eda_pointwise=eda_optimization
            in ("pointwise", "combined", "joined", "gated", "tuned", "stable"),
            lambda_joined_eda=eda_optimization
            in ("joined", "gated", "tuned", "stable"),
            lambda_stable_joined_eda=eda_optimization == "stable",
            lambda_fused_eda_decode=eda_optimization in ("gated", "tuned", "stable"),
            eda_decode_block_v=64
            if eda_optimization
            in ("tiles", "combined", "joined", "gated", "tuned", "stable")
            else 16,
            eda_decode_num_warps=4,
        )
        for key in (
            "eda_decode_block_v_fp32",
            "eda_decode_block_v_fp16",
            "eda_decode_num_warps_fp32",
            "eda_decode_num_warps_fp16",
            "eda_decode_small_block_v",
            "eda_decode_small_num_warps",
            "eda_decode_grid_order",
            "eda_decode_small_grid_order",
        ):
            config.pop(key, None)
        if eda_optimization in ("tuned", "stable"):
            config.update(
                eda_decode_block_v_fp32=128,
                eda_decode_block_v_fp16=128,
                eda_decode_num_warps_fp32=8,
                eda_decode_num_warps_fp16=4,
                eda_decode_grid_order="sequence",
                eda_decode_small_block_v=32,
                eda_decode_small_num_warps=4,
                eda_decode_small_grid_order="head",
            )
        if eda_optimization == "stable":
            config["eda_joined_gemm_tile"] = [32, 64, 128, 4, 3]
        else:
            config.pop("eda_joined_gemm_tile", None)
    config.update(
        projected_routing=False, msa_late_v_decode=False, gdn2_state_storage="fp32"
    )
    output.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name != "config.json":
            (output / path.name).symlink_to(path.resolve())
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--eda-optimization",
        choices=(
            "auto",
            "none",
            "pointwise",
            "tiles",
            "combined",
            "joined",
            "gated",
            "tuned",
            "stable",
        ),
        default="auto",
    )
    args = parser.parse_args()
    print(prepare(args.checkpoint, args.output, args.eda_optimization))


if __name__ == "__main__":
    main()
