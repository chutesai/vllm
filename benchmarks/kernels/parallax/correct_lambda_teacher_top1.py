# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute teacher top1 from saved distributions, preserving the old metric."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from lambda_parity_metrics import teacher_top1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    reference_top = {}
    for lane in ("bf16", "native"):
        path = args.root / ("reference-" + lane)
        reference_top[path.name] = {
            f.stem: torch.load(f, weights_only=True)["logits"][:-1].argmax(-1).tolist()
            for pattern in ("prompt_*.pt", "long_*.pt")
            for f in path.glob(pattern)
        }
    corrections = []
    for path in sorted(args.root.glob("parity-floor-*.json")):
        data = json.loads(path.read_text())
        if (
            data.get("status") != "complete"
            or "teacher_artifacts" not in data
            or data.get("teacher_top1_method") == "argmax_first_token_id"
        ):
            continue
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        ties = changes = 0
        for artifact in data["teacher_artifacts"].values():
            old = artifact["top1"]
            new = [teacher_top1(step) for step in artifact["logprobs"]]
            artifact["legacy_dictionary_top1"] = old
            artifact["top1"] = new
            changes += sum(a != b for a, b in zip(old, new, strict=True))
            ties += sum(
                sum(v == max(step.values()) for v in step.values()) > 1
                for step in artifact["logprobs"]
            )
        for lane, result in data["references"].items():
            result["legacy_dictionary_top1_agreement"] = result["summary"][
                "top1_agreement"
            ]
            for row in result["teacher"]:
                row["legacy_dictionary_top1_agreement"] = row["top1_agreement"]
                top = data["teacher_artifacts"][row["case"]]["top1"]
                expected = reference_top[lane][row["case"]]
                row["top1_agreement"] = sum(
                    a == b for a, b in zip(top, expected, strict=True)
                ) / len(top)
            short = [r for r in result["teacher"] if r["case"].startswith("prompt")]
            result["summary"]["top1_agreement"] = sum(
                r["top1_agreement"] * r["tokens"] for r in short
            ) / sum(r["tokens"] for r in short)
        data["teacher_top1_method"] = "argmax_first_token_id"
        data["teacher_metric_correction"] = {
            "original_artifact_sha256": before,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tied_positions_short_and_long": ties,
            "changed_positions_short_and_long": changes,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")
        corrections.append(
            {
                "path": str(path),
                **data["teacher_metric_correction"],
                "summary": data["references"]["reference-bf16"]["summary"],
            }
        )
    (args.root / "teacher-top1-correction.json").write_text(
        json.dumps(
            {"measurement_label": "MEASURED", "corrections": corrections}, indent=2
        )
        + "\n"
    )
    for row in corrections:
        print(Path(row["path"]).name, row["summary"]["top1_agreement"], flush=True)


if __name__ == "__main__":
    main()
