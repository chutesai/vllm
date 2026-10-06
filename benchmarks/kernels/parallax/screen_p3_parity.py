# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply the P2b teacher/NLL/self floors and retain greedy counts separately."""

import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--receipt", type=Path, required=True)
p.add_argument("--references", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
d = json.loads(a.receipt.read_text())
assert d["status"] == "complete"
summary = next(iter(d["references"].values()))["summary"]
rows = []
for row in d["nll"]:
    data = torch.load(a.references / (row["case"] + ".pt"), weights_only=True)
    ids = data["input_ids"][1:]
    ref = (
        -data["logits"][:-1]
        .float()
        .log_softmax(-1)
        .gather(1, ids[:, None])
        .mean()
        .item()
    )
    rows.append({**row, "reference": ref, "delta": row["mean"] - ref})
short = [r for r in rows if r["case"].startswith("prompt")]
short_delta = sum(r["delta"] * r["tokens"] for r in short) / sum(
    r["tokens"] for r in short
)
limits = {
    "teacher_top1_min": 0.942264,
    "teacher_mean_abs_dlp_max": 0.084353,
    "nll_abs_delta_max": 0.01,
    "self_max": 0.578507,
    "greedy_large_margin_floor_max": 3,
}
checks = {
    "teacher_top1": summary["top1_agreement"] >= limits["teacher_top1_min"],
    "teacher_mean_abs_dlp": summary["mean_abs_dlogprob"]
    <= limits["teacher_mean_abs_dlp_max"],
    "nll_short": abs(short_delta) <= 0.01,
    "nll_each_long": all(
        abs(r["delta"]) <= 0.01 for r in rows if r["case"].startswith("long")
    ),
    "self": d["self_consistency_max"] <= limits["self_max"],
}
result = {
    "measurement_label": "MEASURED",
    "receipt": str(a.receipt),
    "limits": limits,
    "summary": summary,
    "self_max": d["self_consistency_max"],
    "short_nll_delta": short_delta,
    "long_nll": [r for r in rows if r["case"].startswith("long")],
    "checks": checks,
    "requested_teacher_nll_self_screen_pass": all(checks.values()),
    "greedy_count_within_observed_floor": summary["divergences_with_margin_ge_0_1"]
    <= 3,
    "greedy_count_caveat": (
        "eight sequences are statistically underpowered; retain count plainly"
    ),
}
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
