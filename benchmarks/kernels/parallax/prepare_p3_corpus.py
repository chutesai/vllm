# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recover the exact P2b real-text token corpus for matched shadow screens."""

import json
from pathlib import Path

import torch

base = Path("/root/vllm_eda")
ids = [
    torch.load(p, weights_only=True)["input_ids"].tolist()
    for p in sorted((base / "reference-bf16").glob("prompt_*.pt"))
]
(base / "receipts-p3/corpus.json").write_text(json.dumps({"prompts": ids}))
