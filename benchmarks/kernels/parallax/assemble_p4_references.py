# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assemble individually generated mesh traces with source receipt hashes."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--sources", nargs="+", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
rows = []
source_receipts = []
identity = None
for source in a.sources:
    receipt_path = source / "receipt.json"
    d = json.loads(receipt_path.read_text())
    current_identity = {
        key: d[key]
        for key in (
            "manifest_sha256",
            "model_config_sha256",
            "corpus",
            "torch",
            "execution",
        )
    }
    if identity is None:
        identity = current_identity
    else:
        assert (
            current_identity["corpus"]["token_sha256"]
            == identity["corpus"]["token_sha256"]
        )
        for key in ("manifest_sha256", "model_config_sha256", "torch"):
            assert current_identity[key] == identity[key]
        assert (
            current_identity["execution"]["eda_kernel_backend"]
            == identity["execution"]["eda_kernel_backend"]
        )
    source_receipts.append(
        {
            "path": str(receipt_path),
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "status": d["status"],
            "argv": d["argv"],
        }
    )
    for row in d["greedy"]:
        i = int(row["name"].split("_")[-1])
        if any(r["name"] == row["name"] for r in rows):
            continue
        path = source / f"greedy_{i:03d}.pt"
        data = torch.load(path, weights_only=True)
        assert len(data["tokens"]) == 128
        shutil.copyfile(path, a.output / path.name)
        rows.append(
            {
                **row,
                "path": str(a.output / path.name),
                "assembled_from": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
assert {r["name"] for r in rows} == {f"greedy_{i:03d}" for i in range(32)}
(a.output / "receipt.json").write_text(
    json.dumps(
        {
            "status": "complete",
            "measurement_label": "MEASURED",
            "scope": (
                "32 single-sequence mesh repeated-forward traces; "
                "no batched substitution"
            ),
            "identity": identity,
            "sources": source_receipts,
            "greedy": sorted(rows, key=lambda r: r["name"]),
        },
        indent=2,
    )
)
