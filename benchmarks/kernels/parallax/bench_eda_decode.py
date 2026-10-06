# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUPTI graph/cold-L2 sweep of the EDA decode kernel, excluding allocation."""

import json
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.mamba.eda.parallax_eda_decode import _scan, decode
from vllm.triton_utils import triton

rows = []
torch.manual_seed(11)
for batch in (1, 4, 256):
    for dtype in (torch.float32, torch.float16):
        h, d = 9, 128
        q, k, v, e = [
            torch.randn(batch, h, d, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
        q, k, e = [
            torch.nn.functional.normalize(x.float(), dim=-1).bfloat16()
            for x in (q, k, e)
        ]
        g = -torch.rand(batch, h, d, device="cuda") * 0.1
        b, c = [torch.rand(batch, h, device="cuda") for _ in range(2)]
        old = torch.randn(batch, h, d, d, device="cuda", dtype=dtype) * 0.01
        state = old.clone()
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        slots = torch.arange(batch, device="cuda", dtype=torch.int32)
        reset = torch.zeros(batch, device="cuda", dtype=torch.bool)
        inputs = (q, k, v, e, g, b, c)
        expected = decode(*inputs, state, starts, slots, reset)
        expected_state = state.clone()
        out = torch.empty_like(v)
        for bv in (16, 32, 64, 128):
            for warps in (4, 8):

                def run(
                    qi=q,
                    ki=k,
                    vi=v,
                    ei=e,
                    gi=g,
                    bi=b,
                    ci=c,
                    si=state,
                    oi=out,
                    startsi=starts,
                    slotsi=slots,
                    resetsi=reset,
                    batch=batch,
                    h=h,
                    d=d,
                    bv=bv,
                    warps=warps,
                ):
                    _scan[(batch, h, triton.cdiv(d, bv))](
                        qi,
                        ki,
                        vi,
                        ei,
                        gi,
                        bi,
                        ci,
                        si,
                        oi,
                        startsi,
                        slotsi,
                        resetsi,
                        h,
                        d,
                        *si.stride(),
                        128,
                        bv,
                        d**-0.5,
                        True,
                        num_warps=warps,
                        enable_fp_fusion=False,
                    )

                state.copy_(old)
                run()
                torch.testing.assert_close(out, expected, atol=0.001, rtol=0.01)
                torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=0.005)
                for _ in range(25):
                    run()
                torch.accelerator.synchronize()
                times = bench_gpu_time_with_cupti(
                    run,
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                    input_args=inputs + (state, out, starts, slots, reset),
                )
                us = statistics.median(times) * 1000
                rows.append(
                    {
                        "batch": batch,
                        "dtype": str(dtype),
                        "block_v": bv,
                        "num_warps": warps,
                        "us": us,
                        "state_GBps": 2
                        * old.numel()
                        * old.element_size()
                        / (us * 1000),
                    }
                )
                Path("/root/vllm_eda/receipts-p3/profile-scan-sweep.json").write_text(
                    json.dumps(
                        {
                            "measurement_label": "MEASURED",
                            "gpu": torch.cuda.get_device_name(),
                            "scope": "kernel only; mutable state; graph; cold L2",
                            "rows": rows,
                        },
                        indent=2,
                    )
                )
