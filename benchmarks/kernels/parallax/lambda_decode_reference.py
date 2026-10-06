# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sealed bench-lane BF16 reference for the concatenated decode-NLL corpus."""

import json
from pathlib import Path

import torch
from ops.eval.bench_saves import build_model

base = Path("/root/vllm_eda")
export = base / "exports/lambda_242B/exports/242B-tokens_step21067"
paths = sorted((base / "reference-bf16").glob("prompt_*.pt"))
short = [torch.load(p, weights_only=True)["input_ids"].tolist() for p in paths]
model, cfg, dev = build_model(
    export,
    device="cuda",
    model_config=export / "model_config.json",
    eda_kernel_backend="flashqla_eda",
    step=21067,
    expert_format="bf16",
)
output = base / "reference-decode"
output.mkdir(exist_ok=True)
rows = []
with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for j in range(0, len(short), 4):
        ids = sum(short[j : j + 4], [])
        logits = model(torch.tensor([ids], device=dev))[0].float().cpu()
        name = f"windows_{j // 4:02d}"
        torch.save(
            {"input_ids": torch.tensor(ids), "logits": logits}, output / (name + ".pt")
        )
        rows.append({"case": name, "tokens": len(ids)})
(output / "receipt.json").write_text(
    json.dumps(
        {
            "measurement_label": "MEASURED",
            "expert_format": "bf16",
            "build_model": str(export),
            "cases": rows,
        },
        indent=2,
    )
)
