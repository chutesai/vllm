# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare matched cached-decode losses and attach concatenated mesh scores."""

import argparse
import json
from pathlib import Path

import torch

base = Path("/root/vllm_eda")
receipts = base / "receipts-p3"
parser = argparse.ArgumentParser()
parser.add_argument("--label", default="")
parser.add_argument("--receipts", type=Path, default=receipts)
parser.add_argument("--file-prefix", default="")
args = parser.parse_args()
receipts = args.receipts
label = args.label + "-" if args.label else ""
file_prefix = args.file_prefix
lanes = {
    state: json.loads(
        (receipts / f"{file_prefix}decode-nll-{label}{state}.json").read_text()
    )
    for state in ("fp32", "fp16")
}
assert all(x["status"] == "complete" for x in lanes.values())
rows = []
for a, b in zip(lanes["fp32"]["cases"], lanes["fp16"]["cases"], strict=True):
    assert a["case"] == b["case"] and a["input_sha256"] == b["input_sha256"]
    ref = a.get("reference_token_nll")
    if ref is None:
        data = torch.load(
            base / "reference-decode" / (a["case"] + ".pt"), weights_only=True
        )
        prefix = lanes["fp32"]["prefix"]
        ids = data["input_ids"][prefix:]
        ref = (
            -data["logits"][prefix - 1 : -1]
            .float()
            .log_softmax(-1)
            .gather(1, ids[:, None])
            .squeeze(1)
        ).tolist()
    # The first token is predicted by prefill. Exclude it from decode-kernel NLL.
    n = len(ref) - 1
    reference = sum(ref[1:]) / n
    row = {"case": a["case"], "cached_decode_tokens": n, "reference_nll": reference}
    for state, lane in [("fp32", a), ("fp16", b)]:
        row[f"decode_{state}_nll"] = sum(lane["decode_token_nll"][1:]) / n
        row[f"prefill_{state}_nll"] = sum(lane["prefill_token_nll"][1:]) / n
        row[f"decode_{state}_minus_reference"] = row[f"decode_{state}_nll"] - reference
    row["fp16_degradation"] = row["decode_fp16_nll"] - row["decode_fp32_nll"]
    row["fp16_qualified"] = row["fp16_degradation"] <= 0.01
    rows.append(row)
summary = {
    "measurement_label": "MEASURED",
    "threshold_nats": 0.01,
    "first_prediction_excluded": True,
    "cases": rows,
    "fp16_qualified_all_cases": all(r["fp16_qualified"] for r in rows),
}
for group in ("windows", "long", "all"):
    selected = [r for r in rows if group == "all" or r["case"].startswith(group)]
    count = sum(r["cached_decode_tokens"] for r in selected)
    pooled = {"tokens": count}
    for key in (
        "reference_nll",
        "decode_fp32_nll",
        "decode_fp16_nll",
        "prefill_fp32_nll",
        "prefill_fp16_nll",
        "fp16_degradation",
    ):
        pooled[key] = sum(r[key] * r["cached_decode_tokens"] for r in selected) / count
    summary[group] = pooled
(receipts / f"{file_prefix}decode-nll-{label}verdict.json").write_text(
    json.dumps(summary, indent=2)
)
print(json.dumps(summary, indent=2))
