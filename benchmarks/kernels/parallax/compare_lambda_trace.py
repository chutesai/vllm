# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare eager traces, including independently replayed reference inputs."""

import argparse
import json
from pathlib import Path

import torch


def error(a, b):
    diff = (a.squeeze(0).float() - b.squeeze(0).float()).abs()
    return {
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "equal_fraction": float((diff == 0).float().mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--fork", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    ref, fork = [torch.load(path, weights_only=True) for path in (a.reference, a.fork)]
    rows = []
    for i, x in ref["layers"].items():
        y = fork["layers"][i]
        rows.append(
            {
                "layer": int(i),
                "input": error(x["input"], y["input"]),
                "output": error(x["output"], y["output"]),
            }
        )
    receipt = {
        "measurement_label": "MEASURED",
        "reference": a.reference,
        "fork": a.fork,
        "layers": rows,
    }
    if "logits" in fork:
        n = fork["logits"].shape[0]
        receipt["logits"] = error(ref["logits"][0, :n], fork["logits"])
    a.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(rows[:8], indent=2))


if __name__ == "__main__":
    main()
