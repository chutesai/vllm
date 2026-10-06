# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download the pinned public Kappa export with no authentication."""

import json
import os
from pathlib import Path

from vllm.transformers_utils.repo_utils import hf_api

base = Path("/root/vllm_eda")
os.environ["HF_HOME"] = str(base / "cache/huggingface")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ.pop("HF_TOKEN", None)
revision = "15975e88320522284396bfe15c02e533d8b49dc4"
files = hf_api().list_repo_files(
    "chutesai/parallax-8b-kappa", revision=revision, token=False
)
latest_files = [f for f in files if f.lower().endswith("latest.json")]
if latest_files:
    latest_path = hf_api().hf_hub_download(
        "chutesai/parallax-8b-kappa",
        latest_files[0],
        revision=revision,
        local_dir=base / "exports/kappa-snapshot",
        token=False,
    )
    latest = json.loads(Path(latest_path).read_text())
else:
    matches = [
        f.rsplit("/", 1)[0]
        for f in files
        if "step37898/" in f and f.endswith("model_config.json")
    ]
    if len(matches) != 1:
        raise RuntimeError("Missing LATEST.json and ambiguous step37898 export")
    latest = {
        "export_dir": matches[0],
        "deviation": (
            "LATEST.json absent at pinned revision; resolved explicit step37898"
        ),
    }
print(json.dumps(latest), flush=True)
(base / "kappa-download.json").write_text(
    json.dumps({"revision": revision, "latest": latest}, indent=2)
)
export_path = latest.get("export_dir") or latest.get("path") or latest.get("export")
if not isinstance(export_path, str):
    raise RuntimeError("LATEST.json export path needs explicit resolution")
root = Path(
    hf_api().snapshot_download(
        "chutesai/parallax-8b-kappa",
        revision=revision,
        allow_patterns=["LATEST.json", export_path + "/**"],
        local_dir=base / "exports/kappa-snapshot",
        token=False,
    )
)
configs = list((root / export_path).rglob("model_config.json"))
if len(configs) != 1:
    raise RuntimeError(f"Expected one canonical export, found {configs}")
(base / "kappa-export-path.txt").write_text(str(configs[0].parent))
