# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive throughput tables from complete serving and simultaneous DP receipts."""

import json
import statistics
from pathlib import Path

base = Path("/root/vllm_eda/receipts-p3")
rows = []
for path in sorted(base.glob("*.json")):
    if "progress" in path.name:
        continue
    data = json.loads(path.read_text())
    if "rows" not in data or "model_config" not in data:
        continue
    for row in data["rows"]:
        if row["status"] != "complete":
            continue
        trials = row["request_timing_trials"]
        decode = row["output_tokens"] > 1
        rows.append(
            {
                "receipt": path.name,
                "gpu": data.get("cuda_visible_devices"),
                "state": data.get("state_storage"),
                "prompt": row["prompt_tokens"],
                "batch": row["concurrency"],
                "output": row["output_tokens"],
                "rate": statistics.median(
                    t["decode_output_tokens_per_second"] for t in trials
                )
                if decode
                else row["fresh_prefill_tokens_per_second_including_first_output"],
                "mean_itl_ms": statistics.median(
                    t["per_request_mean_itl_seconds"] for t in trials
                )
                * 1000
                if decode
                else None,
                "preemptions": [t.get("preemptions", 0) for t in trials],
                "expert_path": "sparse_ternary_fp4 (components=1)"
                if data["model_config"].get("single_sparse_fp4_components", 0)
                and (
                    row["concurrency"]
                    if decode
                    else row["concurrency"] * row["prompt_tokens"]
                )
                >= 256
                else "packed_bf16",
            }
        )
dp = []
for state in ("fp32", "fp16"):
    paths = [base / f"dp8-{state}-gpu{gpu}.json" for gpu in range(8)]
    if not all(p.exists() for p in paths):
        continue
    lanes = [json.loads(p.read_text()) for p in paths]
    if not all(len(d.get("rows", [])) == 2 for d in lanes):
        continue
    for index, prompt in enumerate((32, 128)):
        trials = [d["rows"][index]["request_timing_trials"] for d in lanes]
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
(base / "p3-summary.json").write_text(
    json.dumps({"measurement_label": "MEASURED", "rows": rows, "dp8": dp}, indent=2)
)
