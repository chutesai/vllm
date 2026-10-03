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


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    config = json.loads((source / "config.json").read_text())
    profile = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "vllm/transformers_utils/configs/parallax_sm120.json"
        ).read_text()
    )
    if config.get("checkpoint_format") != "kappa_bf16" or config.get(
        "random_weights", False
    ):
        raise ValueError("A converted, trained kappa_bf16 checkpoint is required")
    for key in GEOMETRY:
        if config.get(key) != profile[key]:
            raise ValueError(f"Unqualified model geometry: {key}")
    if not (source / "model.safetensors.index.json").is_file():
        raise ValueError("Missing converted safetensors index")
    if source == output or output.exists():
        raise ValueError("Output must be a new directory, separate from the source")
    config.update({key: profile[key] for key in OPTIONS})
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
    args = parser.parse_args()
    print(prepare(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
