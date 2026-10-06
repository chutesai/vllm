# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUPTI graph/cold-L2 sweep of fused EDA decode with correctness checks."""

import hashlib
import json
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.model_executor.layers.mamba.eda.parallax_eda_decode import decode
from vllm.model_executor.layers.mamba.eda.parallax_eda_fused import gates
from vllm.model_executor.layers.mamba.eda.parallax_eda_gated import _decode_gated
from vllm.triton_utils import triton

rows = []
torch.manual_seed(11)
for batch in (1, 4, 256):
    for dtype in (torch.float32, torch.float16):
        h, d = 9, 128
        mixed = torch.randn(batch, 3 * h * d, device="cuda", dtype=torch.bfloat16)
        erase, f = [
            torch.randn(batch, h * d, device="cuda", dtype=mixed.dtype)
            for _ in range(2)
        ]
        b, c = [
            torch.randn(batch, h, device="cuda", dtype=mixed.dtype) for _ in range(2)
        ]
        a = torch.randn(h, device="cuda")
        dt = torch.randn(h * d, device="cuda", dtype=mixed.dtype)
        old = torch.randn(batch, h, d, d, device="cuda", dtype=dtype) * 0.01
        state = old.clone()
        starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        slots = torch.arange(batch, device="cuda", dtype=torch.int32)
        reset = torch.zeros(batch, device="cuda", dtype=torch.bool)
        prepared = gates(mixed, erase, f, b, c, a, dt, h, d)
        expected = decode(*prepared, state, starts, slots, reset)
        expected_state = state.clone()
        debug = [
            torch.empty_like(x)
            for x in (
                prepared[0],
                prepared[1],
                prepared[3],
                prepared[4],
                prepared[5],
                prepared[6],
            )
        ]
        out = torch.empty_like(expected)
        for bv in (16, 32, 64, 128):
            for warps in (4, 8):
                for order in ("sequence", "head"):

                    def run(
                        mixed=mixed,
                        erase=erase,
                        f=f,
                        b=b,
                        c=c,
                        a=a,
                        dt=dt,
                        state=state,
                        out=out,
                        starts=starts,
                        slots=slots,
                        reset=reset,
                        *,
                        batch=batch,
                        h=h,
                        d=d,
                        bv=bv,
                        warps=warps,
                        order=order,
                        debug=debug,
                        dump=False,
                    ):
                        _decode_gated[(batch * h, triton.cdiv(d, bv))](
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
                            batch,
                            mixed.stride(0),
                            erase.stride(0),
                            f.stride(0),
                            b.stride(0),
                            c.stride(0),
                            *state.stride(),
                            128,
                            bv,
                            d**-0.5,
                            order,
                            DUMP=dump,
                            QDEBUG=debug[0],
                            KDEBUG=debug[1],
                            EDEBUG=debug[2],
                            GDEBUG=debug[3],
                            BDEBUG=debug[4],
                            CDEBUG=debug[5],
                            num_warps=warps,
                            enable_fp_fusion=False,
                        )

                    state.copy_(old)
                    run(dump=True)
                    gate_reference = (
                        prepared[0],
                        prepared[1],
                        prepared[3],
                        prepared[4],
                        prepared[5],
                        prepared[6],
                    )
                    for actual_gate, reference_gate in zip(
                        debug, gate_reference, strict=True
                    ):
                        torch.testing.assert_close(
                            actual_gate, reference_gate, atol=0.001, rtol=0.005
                        )
                    conditioned_state = old.clone()
                    conditioned_out = decode(
                        debug[0],
                        debug[1],
                        prepared[2],
                        debug[2],
                        debug[3],
                        debug[4],
                        debug[5],
                        conditioned_state,
                        starts,
                        slots,
                        reset,
                        block_v=bv,
                        num_warps=warps,
                    )
                    diagnostic = {
                        "batch": batch,
                        "dtype": str(dtype),
                        "block_v": bv,
                        "warps": warps,
                        "order": order,
                        "gate_differences": [
                            {
                                "count": int((x != y).sum()),
                                "max": float((x - y).abs().max()),
                            }
                            for x, y in zip(
                                debug,
                                (
                                    prepared[0],
                                    prepared[1],
                                    prepared[3],
                                    prepared[4],
                                    prepared[5],
                                    prepared[6],
                                ),
                                strict=True,
                            )
                        ],
                        "state_max_abs": float(
                            (state.float() - expected_state.float()).abs().max()
                        ),
                        "conditioned_state_max_abs": float(
                            (state.float() - conditioned_state.float()).abs().max()
                        ),
                    }
                    Path(
                        "/root/vllm_eda/receipts-p4/p4-gated-last-correctness.json"
                    ).write_text(json.dumps(diagnostic, indent=2))
                    torch.testing.assert_close(out, expected, atol=0.001, rtol=0.01)
                    torch.testing.assert_close(
                        state, conditioned_state, atol=2e-5, rtol=0.005
                    )
                    torch.testing.assert_close(
                        out, conditioned_out, atol=0.001, rtol=0.01
                    )
                    for _ in range(25):
                        run()
                    torch.accelerator.synchronize()
                    times = bench_gpu_time_with_cupti(
                        run,
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                        input_args=(
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
                        ),
                    )
                    us = statistics.median(times) * 1000
                    rows.append(
                        {
                            "batch": batch,
                            "dtype": str(dtype),
                            "block_v": bv,
                            "num_warps": warps,
                            "grid_order": order,
                            "us": us,
                            "state_GBps": 2
                            * old.numel()
                            * old.element_size()
                            / (us * 1000),
                            "correctness": diagnostic,
                        }
                    )
                    Path("/root/vllm_eda/receipts-p4/p4-gated-sweep.json").write_text(
                        json.dumps(
                            {
                                "measurement_label": "MEASURED",
                                "gpu": torch.cuda.get_device_name(),
                                "scope": (
                                    "kernel only; mutable state; graph; cold L2; "
                                    "state bytes read+write"
                                ),
                                "script_sha256": hashlib.sha256(
                                    Path(__file__).read_bytes()
                                ).hexdigest(),
                                "rows": rows,
                            },
                            indent=2,
                        )
                    )
                    print(rows[-1], flush=True)
