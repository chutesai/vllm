# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create isolated diagnostic and BF16-control views without changing weights."""

import json
from pathlib import Path

base = Path("/root/vllm_eda")
source = base / "lambda-fp4-floor-default"
for label, overrides in [("lambda-p3-bf16", {"single_sparse_fp4_components": 0})]:
    output = base / label
    output.mkdir(exist_ok=False)
    for path in source.iterdir():
        if path.name != "config.json":
            (output / path.name).symlink_to(path.resolve())
    config = json.loads((source / "config.json").read_text())
    config.update(overrides)
    (output / "config.json").write_text(json.dumps(config, indent=2))
