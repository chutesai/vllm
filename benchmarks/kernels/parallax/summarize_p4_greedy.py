# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Judge 32-sequence greedy continuations against measured F2/F3/F5 floors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def load(path):
    if path.is_file():
        doc = json.loads(path.read_text())
        assert doc["status"] == "complete"
        return [
            (doc["greedy"][str(i)]["tokens"], doc["greedy"][str(i)]["top2_margins"])
            for i in range(32)
        ]
    doc = json.loads((path / "receipt.json").read_text())
    assert doc["status"] == "complete"
    result = []
    for i in range(32):
        d = torch.load(path / f"greedy_{i:03d}.pt", weights_only=True)
        assert len(d["tokens"]) == 128
        result.append(
            (
                d["tokens"].tolist(),
                (d["top20_logprobs"][:, 0] - d["top20_logprobs"][:, 1]).tolist(),
            )
        )
    return result


def compare(reference, candidate):
    rows = []
    for i, ((expected, margins), (actual, _)) in enumerate(
        zip(reference, candidate, strict=True)
    ):
        assert len(expected) == len(actual) == 128
        first = next(
            (
                j
                for j, (a, b) in enumerate(zip(expected, actual, strict=True))
                if a != b
            ),
            None,
        )
        rows.append(
            {
                "sequence": i,
                "exact_match": first is None,
                "first_divergence": first,
                "reference_top2_margin": None if first is None else margins[first],
            }
        )
    count = sum(
        r["reference_top2_margin"] is not None and r["reference_top2_margin"] >= 0.1
        for r in rows
    )
    return {
        "sequences": 32,
        "generated_tokens_per_sequence": 128,
        "large_margin_divergences": count,
        "large_margin_divergences_per_32": float(count),
        "fraction": count / 32,
        "exact_matches": sum(r["exact_match"] for r in rows),
        "rows": rows,
    }


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, default=Path("/root/vllm_eda"))
p.add_argument("--candidates", nargs="+", required=True)
a = p.parse_args()
base, out = a.root, a.root / "receipts-p4"
reference = load(base / "reference-p4-bf16-all")
floors = {
    "F2": compare(reference, load(base / "reference-p4-fla-all")),
    "F3": compare(reference, load(base / "reference-p4-batch32")),
    "F5": compare(
        load(out / "p4-floor-single.json"), load(out / "p4-floor-batch32.json")
    ),
}
limit = max(f["large_margin_divergences"] for f in floors.values())
candidates = {}
for name in a.candidates:
    path = out / (
        "p4-floor-batch32.json" if name == "default" else f"p4-{name}-greedy32.json"
    )
    result = compare(reference, load(path))
    result["within_max_floor_count"] = result["large_margin_divergences"] <= limit
    result["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidates[name] = result
receipt = {
    "measurement_label": "MEASURED",
    "threshold_margin": 0.1,
    "floor_max_count": limit,
    "floor_max_count_per_32": float(limit),
    "floors": floors,
    "candidates": candidates,
}
(out / "p4-greedy32-verdict.json").write_text(json.dumps(receipt, indent=2))
print(
    json.dumps(
        {
            "floor_max_count": limit,
            "floors": {k: v["large_margin_divergences"] for k, v in floors.items()},
            "candidates": {
                k: (v["large_margin_divergences"], v["within_max_floor_count"])
                for k, v in candidates.items()
            },
        },
        indent=2,
    )
)
