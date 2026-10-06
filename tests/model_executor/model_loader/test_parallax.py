# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parallax integration contract regressions."""

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.prepare_parallax_profile import prepare
from vllm.model_executor.model_loader.parallax_export import (
    IndexerBundleError,
    PackError,
    _decode_absolute,
    iter_trunk_groups,
    unpack_bundle,
)


def test_fp32_router_fragment_keeps_wire_precision():
    x = torch.tensor([0.100001, -0.000001, 0.100003], dtype=torch.float32)
    raw = x.view(torch.uint8).numpy().tobytes()
    layout = SimpleNamespace(numel=3)
    assert torch.equal(_decode_absolute(layout, raw), x)
    with pytest.raises(PackError):
        _decode_absolute(layout, raw[:-1])


def test_mts_fragment_id_must_match_layout():
    x = torch.tensor([1.5, -2], dtype=torch.bfloat16)
    raw = x.view(torch.uint8).numpy().tobytes()
    header = json.dumps(
        {"fragment_id": 7, "entries": [["x", "torch.bfloat16", [2], len(raw)]]}
    ).encode()
    payload = struct.pack("<4sHI", b"MTS1", 1, len(header)) + header + raw
    layout = SimpleNamespace(numel=2, fragment_id=7, key_order=["x"])
    assert torch.equal(_decode_absolute(layout, payload), x)
    layout.fragment_id = 8
    with pytest.raises(PackError):
        _decode_absolute(layout, payload)


def test_split_trunk_fragment_cannot_have_a_hole():
    layout = SimpleNamespace(numel=2, key_order=["weight::last[1:3]"], shapes=[[1, 2]])
    manifest = {"trunk": {"0": {"version": 1, "digest": "unused"}}}
    coverage = {"tensors": {"weight": {"shape": [1, 3], "dtype": "torch.bfloat16"}}}
    raw = torch.ones(2, dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes()
    with pytest.raises(PackError, match="overlap/hole"):
        list(iter_trunk_groups(manifest, {0: layout}, coverage, lambda _: raw))


def test_indexer_bundle_rejects_modified_payload():
    x = torch.tensor([1.5, -2], dtype=torch.bfloat16)
    body = x.view(torch.uint8).numpy().tobytes()
    header = {
        "schema": "indexer_bundle_v1",
        "step": 12,
        "term": 3,
        "dtype": "torch.bfloat16",
        "entries": [["layer.index_q_down.weight", [1, 2], 2]],
        "checksum": hashlib.blake2b(body, digest_size=16).hexdigest(),
    }
    payload = json.dumps(header).encode() + b"\n" + body
    _, decoded = unpack_bundle(payload)
    assert torch.equal(decoded["layer.index_q_down.weight"], x.reshape(1, 2))
    with pytest.raises(IndexerBundleError, match="checksum"):
        unpack_bundle(payload[:-1] + bytes([payload[-1] ^ 1]))


def test_profile_preserves_checkpoint_policy_and_weights(tmp_path):
    config = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "vllm/transformers_utils/configs/parallax_sm120.json"
        ).read_text()
    )
    config["logit_scale_max"] = 2.75
    config["projected_routing"] = True
    config["gdn2_state_storage"] = "fp16"
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors.index.json").write_text("{}")
    (source / "model-00000.safetensors").write_bytes(b"fixture")
    output = prepare(source, tmp_path / "output")
    installed = json.loads((output / "config.json").read_text())
    assert installed["logit_scale_max"] == 2.75
    assert installed["projected_routing"] is False
    assert installed["gdn2_state_storage"] == "fp32"
    assert (
        output / "model-00000.safetensors"
    ).resolve() == source / "model-00000.safetensors"
    assert json.loads((source / "config.json").read_text()) == config
    with pytest.raises(ValueError, match="new directory"):
        prepare(source, output)
    config["d_model"] = 2048
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="geometry"):
        prepare(source, tmp_path / "invalid")


def test_native_config_preserves_geometry_and_maps_selected_legacy_kernel():
    from vllm.transformers_utils.configs.parallax import ParallaxConfig

    config = ParallaxConfig(
        d_model=2048, single_sparse_fp4_kernel="v28", logit_scale_max=2.75
    )
    assert config.hidden_size == 2048
    assert config.single_sparse_fp4_kernel == "sparse_ternary_fp4"
    assert config.logit_scale_max == 2.75


@pytest.mark.parametrize(
    "options",
    [
        {"single_sparse_fp4_kernel": "v26"},
        {"expert_backend": "grouped_ternary"},
        {"projected_routing": True},
    ],
)
def test_native_config_rejects_retired_unshipped_execution_paths(options):
    from vllm.transformers_utils.configs.parallax import ParallaxConfig

    with pytest.raises(ValueError):
        ParallaxConfig(**options)


@pytest.mark.parametrize(
    "optimization",
    [
        "none",
        "pointwise",
        "tiles",
        "combined",
        "joined",
        "gated",
        "tuned",
        "stable",
        "auto",
    ],
)
def test_lambda_profile_keeps_dense_attention_and_eda_geometry(tmp_path, optimization):
    root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (root / "vllm/transformers_utils/configs/parallax_sm120.json").read_text()
    )
    config.update(
        checkpoint_format="lambda_bf16",
        recurrent_backend="eda",
        hybrid_override_pattern="WERERERE*EREWERERE*EWEREREWE*EREREWERE*EREWERERE*EWERERERE*EREWE",
        tie_word_embeddings=True,
        eda_erase_rank=16,
        eda_decay_rank=None,
        eda_gate_lower=-5.0,
        eda_onorm_eps=1e-6,
        eda_scale_mode="dk^-0.5",
        n_shared_experts=1,
        final_norm_unit_gain=True,
        moe_router_fp32=True,
        sliding_window_size=2048,
        logit_scale_max=3.0,
        pinned_swa_windows=[[0, 2048], [62, 2048]],
        msa_index_query_chunk=256,
    )
    source = tmp_path / "lambda"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors.index.json").write_text("{}")
    output = prepare(source, tmp_path / "profile", optimization)
    optimization = "stable" if optimization == "auto" else optimization
    installed = json.loads((output / "config.json").read_text())
    assert installed["inference_attention_mode"] == "dense"
    assert installed["joined_bf16_gdn2"] is False
    assert installed["head_dtype"] == "bfloat16"
    assert installed["lambda_fused_eda_pointwise"] == (
        optimization in ("pointwise", "combined", "joined", "gated", "tuned", "stable")
    )
    assert installed["lambda_joined_eda"] == (
        optimization in ("joined", "gated", "tuned", "stable")
    )
    assert installed["lambda_fused_eda_decode"] == (
        optimization in ("gated", "tuned", "stable")
    )
    assert installed["lambda_stable_joined_eda"] == (optimization == "stable")
    assert installed["eda_decode_block_v"] == (
        64
        if optimization in ("tiles", "combined", "joined", "gated", "tuned", "stable")
        else 16
    )
    if optimization in ("tuned", "stable"):
        assert installed["eda_decode_block_v_fp32"] == 128
        assert installed["eda_decode_num_warps_fp32"] == 8
        assert installed["eda_decode_num_warps_fp16"] == 4
        assert installed["eda_decode_small_grid_order"] == "head"
    if optimization == "stable":
        assert installed["eda_joined_gemm_tile"] == [32, 64, 128, 4, 3]
    for key in (
        "hybrid_override_pattern",
        "pinned_swa_windows",
        "msa_index_query_chunk",
        "eda_erase_rank",
        "logit_scale_max",
        "tie_word_embeddings",
    ):
        assert installed[key] == config[key]
    config["eda_erase_rank"] = 8
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="eda_erase_rank"):
        prepare(source, tmp_path / "bad")


@pytest.mark.parametrize("mesh_norm", [False, True])
def test_lambda_norm_loader_preserves_unit_gain_and_bounded_scale(mesh_norm):
    """Selecting upstream norm must still apply the export's normalization policy."""
    from vllm.model_executor.model_loader.parallax import load_kappa_weights

    model = torch.nn.Module()
    model.config = SimpleNamespace(
        random_weights=False,
        recurrent_backend="eda",
        lambda_mesh_norm=mesh_norm,
        final_norm_unit_gain=True,
        hybrid_override_pattern="",
        n_routed_experts=0,
        logit_scale_max=3.0,
    )
    model.layers = torch.nn.ModuleList()
    model.final_norm = torch.nn.Module()
    model.final_norm.weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
    model.register_buffer("logit_scale", torch.ones(1, dtype=torch.float32))
    weight = torch.tensor([0.912345, 1.012345, 1.112345, 1.212345])
    load_kappa_weights(
        model,
        [
            ("final_norm.weight", weight),
            ("final_norm.logit_scale_log", torch.tensor([1.1015625])),
        ],
    )
    if mesh_norm:
        expected = weight
    else:
        rounded = weight.bfloat16().float()
        expected = (
            (rounded / rounded.square().mean().add(1e-6).sqrt()).bfloat16().float()
        )
    assert model.final_norm.weight.dtype == torch.float32
    torch.testing.assert_close(model.final_norm.weight, expected, atol=0, rtol=0)
    torch.testing.assert_close(model.logit_scale, torch.tensor([3.0]))
