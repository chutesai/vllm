# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute Kineto GPU kernel duration and diagnostic module events."""

import argparse
import collections
import gzip
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
rows = []
for path in a.directory.rglob("*.pt.trace.json*"):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    totals = collections.defaultdict(lambda: [0, 0.0])
    for e in data["traceEvents"]:
        if e.get("cat") == "kernel" and "dur" in e:
            item = totals[e["name"]]
            item[0] += 1
            item[1] += e["dur"]
    rows.append(
        {
            "trace": str(path),
            "kernels": sorted(
                [
                    {"name": n, "calls": v[0], "gpu_ms": v[1] / 1000}
                    for n, v in totals.items()
                ],
                key=lambda x: -x["gpu_ms"],
            ),
        }
    )
a.output.write_text(
    json.dumps(
        {
            "measurement_label": "MEASURED",
            "scope": (
                "sum of GPU kernel duration in diagnostic profiled generation; "
                "not wall time"
            ),
            "traces": rows,
        },
        indent=2,
    )
)
