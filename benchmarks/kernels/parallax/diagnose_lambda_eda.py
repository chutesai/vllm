# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay real layer inputs with identical projections to isolate EDA kernels."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.model_executor.layers.mamba.eda.parallax_eda_decode import decode
from vllm.model_executor.layers.mamba.eda.parallax_eda_pointwise import gates, norm_gate
from vllm.model_executor.layers.mamba.eda.parallax_eda_prefill import prefill


def error(a, b):
    diff = (a.float() - b.float()).abs()
    return {"max": float(diff.max()), "mean": float(diff.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    config = json.loads((a.checkpoint / "config.json").read_text())
    index = json.loads((a.checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    data = torch.load(a.trace, weights_only=True)
    rows = []
    for i, kind in enumerate(config["hybrid_override_pattern"]):
        if kind != "R":
            continue

        def load(suffix, norm=False, layer_index=i):
            name = f"layers.{layer_index}." + ("" if norm else "_fast_impl.") + suffix
            with safe_open(a.checkpoint / index[name], framework="pt") as f:
                return f.get_tensor(name).cuda()

        x = data["teacher"][f"layer.{i}"][0]["input"].cuda()
        xf = x.float()
        x = (
            xf
            * xf.square().mean(-1, keepdim=True).add(1e-6).rsqrt()
            * load("norm.weight", norm=True)
        ).to(x.dtype)

        def linear(suffix, value=x, bias=False):
            return F.linear(
                value,
                load(suffix + ".weight"),
                load(suffix + ".bias") if bias else None,
            )

        mixed = []
        for name in ("q", "k", "v"):
            projected = linear(name + "_proj")
            conv = F.conv1d(
                projected.T.unsqueeze(0),
                load(name + "_conv1d.weight"),
                padding=3,
                groups=1152,
            )[..., : x.shape[0]]
            mixed.append(F.silu(conv).squeeze(0).T)
        erase = linear("e_proj.1", linear("e_proj.0"))
        f = linear("f_proj.1", linear("f_proj.0"))
        xs = gates(
            torch.cat(mixed, -1),
            erase,
            f,
            linear("b_proj", bias=True),
            linear("c_proj", bias=True),
            load("A_log"),
            load("dt_bias"),
            9,
            128,
        )
        state = torch.zeros(1, 9, 128, 128, device="cuda")
        continued = state.clone()
        slot = torch.tensor([0], device="cuda", dtype=torch.int32)
        reset = torch.tensor([True], device="cuda")
        starts = torch.tensor([0, x.shape[0]], device="cuda", dtype=torch.int32)
        full = prefill(*xs, state, starts, slot, reset)
        prefill(
            *[z[:128] for z in xs], continued, starts.new_tensor([0, 128]), slot, reset
        )
        reset.fill_(False)
        outputs = []
        for t in range(128, x.shape[0]):
            outputs.append(
                decode(
                    *[z[t : t + 1] for z in xs],
                    continued,
                    starts.new_tensor([0, 1]),
                    slot,
                    reset,
                )
            )
        actual = torch.cat(outputs)
        gate = linear("g_proj.1", linear("g_proj.0"), bias=True).reshape(-1, 9, 128)
        expected_branch = linear(
            "o_proj",
            norm_gate(full[128:], gate[128:], load("o_norm.weight")).flatten(1),
        )
        actual_branch = linear(
            "o_proj", norm_gate(actual, gate[128:], load("o_norm.weight")).flatten(1)
        )
        rows.append(
            {
                "layer": i,
                "recurrent_output": error(actual, full[128:]),
                "branch_output": error(actual_branch, expected_branch),
                "final_state": error(continued, state),
            }
        )
    a.output.write_text(
        json.dumps(
            {
                "measurement_label": "MEASURED",
                "trace": str(a.trace),
                "scope": (
                    "Identical prepared real inputs: "
                    "full FlashQLA vs prefix plus decode"
                ),
                "layers": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
