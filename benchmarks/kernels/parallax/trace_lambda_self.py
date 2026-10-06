# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locate the first fork prefill/decode difference on an identical token prefix."""

import argparse
import json
from pathlib import Path

import torch

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    ids = json.loads(Path(__file__).with_name("verification_inputs.json").read_text())[
        "prompts"
    ][0]
    tracepath = a.output.with_suffix(".pt")
    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.7,
        worker_extension_cls="vllm.v1.worker.gpu.parallax_trace.LambdaTraceWorker",
    )
    try:
        llm.collective_rpc("lambda_install_self_trace", args=(str(tracepath),))
        output = llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(
                temperature=0,
                max_tokens=4,
                ignore_eos=True,
                detokenize=False,
                logprobs=0,
            ),
            use_tqdm=False,
        )[0]
        generated = list(output.outputs[0].token_ids)
        llm.collective_rpc("lambda_mark_self_teacher")
        llm.generate(
            [{"prompt_token_ids": ids + generated}],
            SamplingParams(
                temperature=0,
                max_tokens=1,
                ignore_eos=True,
                detokenize=False,
                prompt_logprobs=0,
            ),
            use_tqdm=False,
        )
        llm.collective_rpc("lambda_save_self_trace")
    finally:
        llm.llm_engine.engine_core.shutdown()
    data = torch.load(tracepath, weights_only=True)
    rows = []
    for name, steps in data["decode"].items():
        row = {"module": name}
        for kind in ("input", "output"):
            actual = torch.cat([s[kind] for s in steps]).float()
            expected = data["teacher"][name][0][kind][
                len(ids) : len(ids) + len(steps)
            ].float()
            diff = (actual - expected).abs()
            row[kind] = {
                "max": float(diff.max()),
                "mean": float(diff.mean()),
                "equal_fraction": float((diff == 0).float().mean()),
            }
        rows.append(row)
    receipt = {
        "measurement_label": "MEASURED",
        "trace": str(tracepath),
        "layers": rows,
        "generated": generated,
    }
    a.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(rows[:4], indent=2))


if __name__ == "__main__":
    main()
