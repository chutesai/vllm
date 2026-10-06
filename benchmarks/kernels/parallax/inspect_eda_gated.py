# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve generated code and resource metadata for selected EDA decode tiles."""

import hashlib
import json
from pathlib import Path

import torch

from vllm.model_executor.layers.mamba.eda.parallax_eda_gated import _decode_gated

outdir = Path("/root/vllm_eda/receipts-p4/generated")
outdir.mkdir(parents=True, exist_ok=True)
rows = []
for dtype, warps in ((torch.float32, 8), (torch.float16, 4)):
    h, d, n = 9, 128, 256
    mixed = torch.randn(n, 3 * h * d, device="cuda", dtype=torch.bfloat16)
    erase, f = [
        torch.randn(n, h * d, device="cuda", dtype=mixed.dtype) for _ in range(2)
    ]
    b, c = [torch.randn(n, h, device="cuda", dtype=mixed.dtype) for _ in range(2)]
    a = torch.randn(h, device="cuda")
    dt = torch.randn(h * d, device="cuda", dtype=mixed.dtype)
    state = torch.zeros(n, h, d, d, device="cuda", dtype=dtype)
    out = torch.empty(n, h, d, device="cuda", dtype=mixed.dtype)
    starts = torch.arange(n + 1, device="cuda", dtype=torch.int32)
    slots = torch.arange(n, device="cuda", dtype=torch.int32)
    reset = torch.zeros(n, device="cuda", dtype=torch.bool)
    kernel = _decode_gated[(n * h, 1)](
        mixed,
        erase,
        f,
        b,
        c,
        a,
        dt,
        state,
        out,
        starts,
        slots,
        reset,
        h,
        d,
        n,
        mixed.stride(0),
        erase.stride(0),
        f.stride(0),
        b.stride(0),
        c.stride(0),
        *state.stride(),
        128,
        128,
        d**-0.5,
        "sequence",
        num_warps=warps,
        enable_fp_fusion=False,
    )
    name = str(dtype).split(".")[-1]
    for kind in ("ptx", "sass", "ttgir"):
        if kind in kernel.asm:
            path = outdir / f"p4-gated-{name}.{kind}"
            path.write_text(kernel.asm[kind])
    ptx = kernel.asm["ptx"]
    rows.append(
        {
            "state_dtype": str(dtype),
            "block_v": 128,
            "num_warps": warps,
            "num_registers": kernel.n_regs,
            "num_spills": kernel.n_spills,
            "shared_bytes": kernel.metadata.shared,
            "local_load_instructions": ptx.count("ld.local"),
            "local_store_instructions": ptx.count("st.local"),
            "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
        }
    )
Path("/root/vllm_eda/receipts-p4/p4-generated-code.json").write_text(
    json.dumps(
        {
            "measurement_label": "MEASURED",
            "scope": "compiler resources and generated PTX for selected tiles",
            "gpu": torch.cuda.get_device_name(),
            "rows": rows,
        },
        indent=2,
    )
)
