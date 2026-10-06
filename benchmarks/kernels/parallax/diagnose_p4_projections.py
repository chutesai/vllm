# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate joined-input and batched-expansion rounding on real saved activations."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.model_executor.layers.mamba.eda.parallax_eda_linear import (
    StableJoinedEDAProjection,
)
from vllm.model_executor.layers.parallax_linear import join_linears

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--trace", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
cfg = json.loads((a.checkpoint / "config.json").read_text())
index = json.loads((a.checkpoint / "model.safetensors.index.json").read_text())[
    "weight_map"
]
trace = torch.load(a.trace, weights_only=True)
rows = []
for i, kind in enumerate(cfg["hybrid_override_pattern"]):
    if kind != "R":
        continue

    def load(suffix, layer_index=i):
        name = f"layers.{layer_index}.{suffix}"
        with safe_open(a.checkpoint / index[name], framework="pt") as f:
            return f.get_tensor(name).to(device="cuda", dtype=torch.bfloat16)

    raw = trace["layers"][str(i)]["input"].reshape(-1, cfg["d_model"]).cuda()
    x = F.rms_norm(
        raw.float(), (cfg["d_model"],), load("norm.weight").float(), 1e-6
    ).bfloat16()
    modules = []
    for suffix, bias in (
        ("q_proj", False),
        ("k_proj", False),
        ("v_proj", False),
        ("e_proj.0", False),
        ("f_proj.0", False),
        ("b_proj", True),
        ("c_proj", True),
        ("g_proj.0", False),
    ):
        w = load("_fast_impl." + suffix + ".weight")
        layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=bias, device="meta")
        layer.weight = torch.nn.Parameter(w, requires_grad=False)
        if bias:
            layer.bias = torch.nn.Parameter(
                load("_fast_impl." + suffix + ".bias"), requires_grad=False
            )
        modules.append(layer)
    joined = join_linears(modules)
    stable = StableJoinedEDAProjection.from_joined(joined)
    e1, f1 = [
        load("_fast_impl." + suffix + ".weight") for suffix in ("e_proj.1", "f_proj.1")
    ]
    ef_weight = torch.stack((e1.T, f1.T)).contiguous()
    for batch in (1, 4, 32, 128, 256):
        value = x.repeat((2, 1))[:batch]
        originals = [m(value) for m in modules]
        pieces = joined(value).split([m.out_features for m in modules], -1)
        stable_pieces = stable(value).split([m.out_features for m in modules], -1)
        exact = [
            F.linear(
                value.float(),
                m.weight.float(),
                m.bias.float() if m.bias is not None else None,
            ).bfloat16()
            for m in modules
        ]
        unbatched = (F.linear(originals[3], e1), F.linear(originals[4], f1))
        batched = torch.bmm(
            torch.stack((originals[3], originals[4])), ef_weight
        ).unbind(0)

        def difference(left, right):
            return {
                "elements": left.numel(),
                "different": int((left != right).sum()),
                "max_abs": float((left.float() - right.float()).abs().max()),
            }

        rows.append(
            {
                "layer": i,
                "batch": batch,
                "joined_input": [
                    difference(u, v) for u, v in zip(originals, pieces, strict=True)
                ],
                "batched_expansion": [
                    difference(u, v) for u, v in zip(unbatched, batched, strict=True)
                ],
                "stable_joined_input": [
                    difference(u, v)
                    for u, v in zip(originals, stable_pieces, strict=True)
                ],
                "original_vs_fp32_oracle": [
                    difference(u, v) for u, v in zip(originals, exact, strict=True)
                ],
                "stable_vs_fp32_oracle": [
                    difference(u, v) for u, v in zip(stable_pieces, exact, strict=True)
                ],
            }
        )
a.output.write_text(
    json.dumps(
        {
            "measurement_label": "MEASURED",
            "trace": str(a.trace),
            "trace_sha256": hashlib.sha256(a.trace.read_bytes()).hexdigest(),
            "rows": rows,
        },
        indent=2,
    )
)
