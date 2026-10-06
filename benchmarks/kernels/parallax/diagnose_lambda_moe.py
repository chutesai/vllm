# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconstruct one saved mesh MoE layer to isolate grouped-kernel rounding."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--torch-only", action="store_true")
    p.add_argument("--full-accum", action="store_true")
    args = p.parse_args()
    if args.full_accum:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"layers.{args.layer}."

    def load(suffix):
        name = prefix + suffix
        with safe_open(args.checkpoint / index[name], framework="pt") as f:
            return f.get_tensor(name).cuda()

    trace = torch.load(args.trace, weights_only=True)
    data = trace["layers"][str(args.layer)]
    x = data["input"].squeeze(0).cuda()
    h = F.rms_norm(x, (x.shape[-1],), load("norm.weight").to(x.dtype), 1e-6)
    logits = F.linear(h.float(), load("router.gate.weight"))
    ids = (logits - load("router.qb_beta")).topk(12, -1).indices
    selected = logits.bfloat16().sigmoid().gather(1, ids)
    gates = selected.float() / selected.float().sum(-1, keepdim=True).clamp_min(1e-6)
    latent = F.linear(h, load("latent_down.weight"))
    up = torch.stack([load(f"experts.{e}.up_proj.weight") for e in range(128)])
    down = torch.stack([load(f"experts.{e}.down_proj.weight") for e in range(128)])
    ref = torch.zeros_like(latent)
    for expert in range(128):
        tokens, routes = (ids == expert).nonzero(as_tuple=True)
        if tokens.numel():
            value = F.linear(
                F.linear(latent[tokens], up[expert]).relu().square(), down[expert]
            )
            ref[tokens] += value * gates[tokens, routes, None]
    scales = torch.ones(128, device="cuda")
    if not args.torch_only:
        from vllm.model_executor.layers.fused_moe.parallax.packed_bf16 import moe

        kernel = moe(
            latent,
            up,
            down,
            scales,
            scales,
            ids,
            gates.float(),
            packed=False,
            mesh_bf16_reduce=True,
        )
    else:
        kernel = ref
    shared = F.linear(
        F.linear(h, load("shared_experts.0.up_proj.weight")).relu().square(),
        load("shared_experts.0.down_proj.weight"),
    )
    expected = x + (F.linear(ref, load("latent_up.weight")) + shared)
    actual = x + (F.linear(kernel, load("latent_up.weight")) + shared)

    def err(a, b):
        diff = (a.float() - b.float()).abs()
        return {
            "max": float(diff.max()),
            "mean": float(diff.mean()),
            "equal_fraction": float((diff == 0).float().mean()),
        }

    receipt = {
        "measurement_label": "MEASURED",
        "torch": torch.__version__,
        "full_accum_control": args.full_accum,
        "layer": args.layer,
        "torch_reconstruction_vs_mesh": err(expected, data["output"].squeeze(0).cuda()),
        "grouped_vs_torch_routed": err(kernel, ref),
        "grouped_reconstruction_vs_mesh": err(actual, data["output"].squeeze(0).cuda()),
    }
    if "moe_components" in trace:
        components = trace["moe_components"]
        receipt["components"] = {
            "norm": err(h, components["norm"]["output"].squeeze(0).cuda()),
            "latent_down": err(latent, components["latent_down"]["output"].cuda()),
            "router_logits_bf16": err(
                logits.bfloat16(), components["router"]["output"][2].cuda()
            ),
            "router_gate": err(gates, components["router"]["output"][1].cuda()),
            "router_indices_equal": torch.equal(
                ids, components["router"]["output"][0].cuda()
            ),
            "routed_torch": err(ref, components["latent_up"]["input"].cuda()),
            "routed_kernel": err(kernel, components["latent_up"]["input"].cuda()),
            "shared": err(
                shared, components["shared_experts.0.down_proj"]["output"].cuda()
            ),
        }
        receipt["weight_mismatches"] = {}
        for name, weight in trace["moe_weights"].items():
            canonical = load(name)
            if not torch.equal(canonical.cpu(), weight):
                receipt["weight_mismatches"][name] = err(canonical, weight.cuda())
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
