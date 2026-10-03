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
