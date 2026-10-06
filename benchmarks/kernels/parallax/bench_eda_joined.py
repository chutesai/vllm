# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUPTI sweep of precise joined EDA projections with a FP32 GEMM oracle."""

import json
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.mamba.eda.parallax_eda_linear import _linear
from vllm.triton_utils import triton

torch.manual_seed(67)
torch.set_float32_matmul_precision("highest")
rows = []
n, k = 3890, 1152
weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
bias = torch.zeros(n, device="cuda", dtype=weight.dtype)
bias[3744:3762] = torch.randn(18, device="cuda", dtype=weight.dtype) * 0.1
for m in (1, 4, 32, 128, 256):
    x = torch.randn(m + 1, k, device="cuda", dtype=weight.dtype)[1:]
    expected = torch.nn.functional.linear(
        x.float(), weight.float(), bias.float()
    ).bfloat16()
    out = torch.empty_like(expected)

    def native(x=x, weight=weight, bias=bias, out=out):
        torch.addmm(bias, x, weight.T, out=out)

    native()
    native_error = {
        "different": int((out != expected).sum()),
        "max_abs": float((out.float() - expected.float()).abs().max()),
    }
    native_times = bench_gpu_time_with_cupti(
        native,
        use_cuda_graph=True,
        cold_l2_cache=True,
        input_args=(x, weight, bias, out),
    )
    rows.append(
        {
            "m": m,
            "implementation": "torch.addmm",
            "us": statistics.median(native_times) * 1000,
            "correctness": native_error,
        }
    )
    for bm, bn, bk, warps in (
        (16, 64, 64, 4),
        (32, 64, 64, 4),
        (32, 128, 64, 4),
        (64, 64, 64, 4),
        (64, 128, 64, 4),
        (32, 64, 128, 4),
        (32, 128, 128, 4),
        (64, 128, 64, 8),
    ):

        def run(
            x=x,
            weight=weight,
            bias=bias,
            out=out,
            *,
            m=m,
            bm=bm,
            bn=bn,
            bk=bk,
            warps=warps,
        ):
            _linear[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
                x,
                weight,
                bias,
                out,
                m,
                n,
                k,
                x.stride(0),
                weight.stride(0),
                bm,
                bn,
                bk,
                num_warps=warps,
                num_stages=3,
                enable_fp_fusion=False,
            )

        run()
        torch.testing.assert_close(out, expected, atol=0.001, rtol=0.01)
        error = {
            "different": int((out != expected).sum()),
            "max_abs": float((out.float() - expected.float()).abs().max()),
        }
        times = bench_gpu_time_with_cupti(
            run,
            use_cuda_graph=True,
            cold_l2_cache=True,
            input_args=(x, weight, bias, out),
        )
        rows.append(
            {
                "m": m,
                "implementation": "triton_fp32_acc",
                "tile": [bm, bn, bk, warps, 3],
                "us": statistics.median(times) * 1000,
                "correctness": error,
            }
        )
        Path("/root/vllm_eda/receipts-p4/p4-joined-gemm-sweep.json").write_text(
            json.dumps(
                {
                    "measurement_label": "MEASURED",
                    "scope": "kernel only; graph; cold L2; allocations excluded",
                    "gpu": torch.cuda.get_device_name(),
                    "rows": rows,
                },
                indent=2,
            )
        )
        print(rows[-1], flush=True)
