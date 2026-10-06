# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write NLL receipts from the measured floor/parity artifacts."""

import argparse
import json
from pathlib import Path

from analyze_lambda_floor import nll_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    reference = json.loads((args.root / "reference-bf16/receipt.json").read_text())
    ref_rows = [
        {"case": r["name"], "tokens": r["length"] - 1, "mean": r["loss"]}
        for r in reference["teacher"]
    ]
    ref_mean = nll_summary(ref_rows)
    for path in sorted(args.root.glob("parity-floor-*.json")):
        data = json.loads(path.read_text())
        if data.get("status") != "complete" or "references" not in data:
            continue
        rows = data["nll"]
        mean = nll_summary(rows)
        delta = {k: mean[k] - ref_mean[k] for k in mean}
        name = path.stem.removeprefix("parity-floor-")
        nll = {
            "measurement_label": "MEASURED",
            "status": "complete",
            "source": str(path),
            "reference": str(args.root / "reference-bf16/receipt.json"),
            "cases": rows,
            "mean": mean,
            "reference_mean": ref_mean,
            "delta": delta,
            "per_long_delta": {
                r["case"]: r["mean"]
                - next(x["mean"] for x in ref_rows if x["case"] == r["case"])
                for r in rows
                if r["case"].startswith("long")
            },
        }
        (args.root / ("nll-" + name + ".json")).write_text(
            json.dumps(nll, indent=2) + "\n"
        )
        s = data["references"]["reference-bf16"]["summary"]
        print(
            name,
            json.dumps(s),
            "self",
            data["self_consistency_max"],
            "nll_delta",
            delta,
            flush=True,
        )
    for lane in ("bf16", "native", "fla", "batch32"):
        path = args.root / ("reference-" + lane) / "receipt.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if len(data["teacher"]) != 34:
            continue
        rows = [
            {"case": r["name"], "tokens": r["length"] - 1, "mean": r["loss"]}
            for r in data["teacher"]
        ]
        mean = nll_summary(rows)
        nll = {
            "measurement_label": "MEASURED",
            "status": "complete",
            "source": str(path),
            "cases": rows,
            "mean": mean,
            "delta": {k: mean[k] - ref_mean[k] for k in mean},
        }
        (args.root / ("nll-reference-" + lane + ".json")).write_text(
            json.dumps(nll, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
