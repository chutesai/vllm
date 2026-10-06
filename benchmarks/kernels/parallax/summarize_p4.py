# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive P4 serving and cohort rates from completed real-inference receipts."""

import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, default=Path("/root/vllm_eda/receipts-p4"))
a = p.parse_args()
rows = []
for path in sorted(a.root.glob("p4-*.json")):
    if "progress" in path.name:
        continue
    d = json.loads(path.read_text())
    if "model_config" not in d or "rows" not in d:
        continue
    for r in d["rows"]:
        if r["status"] != "complete":
            continue
        decode = r["output_tokens"] > 1
        ts = r["request_timing_trials"]
        rows.append(
            {
                "receipt": path.name,
                "gpu": d["cuda_visible_devices"],
                "state": d["state_storage"],
                "prompt": r["prompt_tokens"],
                "batch": r["concurrency"],
                "output": r["output_tokens"],
                "rate": statistics.median(
                    t["decode_output_tokens_per_second"] for t in ts
                )
                if decode
                else r["fresh_prefill_tokens_per_second_including_first_output"],
                "mean_itl_ms": statistics.median(
                    t["per_request_mean_itl_seconds"] for t in ts
                )
                * 1000
                if decode
                else None,
                "preemptions": [t.get("preemptions", 0) for t in ts],
            }
        )
dp = []
for state in ("fp32", "fp16"):
    paths = [a.root / f"p4-final-dp8-{state}-gpu{gpu}.json" for gpu in range(8)]
    if not all(path.exists() for path in paths):
        continue
    lanes = [json.loads(path.read_text()) for path in paths]
    assert all(len(d.get("rows", [])) == 2 for d in lanes)
    for index, prompt in enumerate((32, 128)):
        trials = [d["rows"][index]["request_timing_trials"] for d in lanes]
        assert all(d["rows"][index]["status"] == "complete" for d in lanes)
        cohort = [
            8
            * 256
            * 127
            / (
                max(t[j]["last_token_ts"] for t in trials)
                - min(t[j]["first_token_ts"] for t in trials)
            )
            for j in range(5)
        ]
        per_gpu = [
            statistics.median(t["decode_output_tokens_per_second"] for t in ts)
            for ts in trials
        ]
        dp.append(
            {
                "state": state,
                "prompt": prompt,
                "per_gpu": per_gpu,
                "sum_per_gpu_medians": sum(per_gpu),
                "cohort_decode_rate_median": statistics.median(cohort),
                "cohort_decode_rate_trials": cohort,
                "definition": (
                    "8*256*127 / (latest last token - earliest first token), "
                    "simultaneous trial"
                ),
            }
        )
(a.root / "p4-summary.json").write_text(
    json.dumps({"measurement_label": "MEASURED", "rows": rows, "dp8": dp}, indent=2)
)
