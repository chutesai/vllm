# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure forced cached decode against full-prefill and sealed mesh logits."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--state", choices=("fp32", "fp16"), default="fp32")
    p.add_argument("--prefix", type=int, default=64)
    p.add_argument("--window-references", type=Path)
    args = p.parse_args()
    paths = sorted(args.references.glob("prompt_*.pt"))
    short = [torch.load(x, weights_only=True)["input_ids"].tolist() for x in paths]
    cases = [
        (f"windows_{j // 4:02d}", sum(short[j : j + 4], []), None)
        for j in range(0, len(short), 4)
    ]
    for path in sorted(args.references.glob("long_*.pt")):
        data = torch.load(path, weights_only=True)
        cases.append((path.stem, data["input_ids"].tolist(), data["logits"]))
    receipt = {
        "measurement_label": "MEASURED",
        "state": args.state,
        "model": args.model,
        "prefix": args.prefix,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tp": 1,
        "mode": "eager, one request, raw_logprobs, forced targets",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
        "status": "running",
    }
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        enforce_eager=True,
        gpu_memory_utilization=0.7,
        hf_overrides={"gdn2_state_storage": args.state},
        logprobs_mode="raw_logprobs",
        logits_processors=["lambda_force_tokens:ForceTokens"],
    )
    receipt["effective_config"] = (
        llm.llm_engine.vllm_config.model_config.hf_config.to_dict()
    )
    try:
        for name, ids, reference in cases:
            if reference is None and args.window_references:
                data = torch.load(
                    args.window_references / (name + ".pt"), weights_only=True
                )
                assert data["input_ids"].tolist() == ids
                reference = data["logits"]
            targets = ids[args.prefix :]
            full = llm.generate(
                [{"prompt_token_ids": ids}],
                SamplingParams(
                    max_tokens=1,
                    temperature=0,
                    detokenize=False,
                    ignore_eos=True,
                    prompt_logprobs=1,
                ),
                use_tqdm=False,
            )[0]
            forced = llm.generate(
                [{"prompt_token_ids": ids[: args.prefix]}],
                SamplingParams(
                    max_tokens=len(targets),
                    min_tokens=len(targets),
                    temperature=0,
                    detokenize=False,
                    ignore_eos=True,
                    logprobs=1,
                    extra_args={"forced_tokens": targets},
                ),
                use_tqdm=False,
            )[0].outputs[0]
            assert list(forced.token_ids) == targets
            decode = [
                -step[token].logprob
                for step, token in zip(forced.logprobs, targets, strict=True)
            ]
            prefill = [
                -full.prompt_logprobs[j][ids[j]].logprob
                for j in range(args.prefix, len(ids))
            ]
            # A processed forced distribution has logprob zero. Raw distributions
            # must retain nonzero loss and match the unforced first-token score.
            assert sum(decode) > 0
            assert abs(decode[0] - prefill[0]) < 0.6
            row = {
                "case": name,
                "tokens": len(targets),
                "decode_nll": sum(decode) / len(decode),
                "cached_decode_nll": sum(decode[1:]) / len(decode[1:]),
                "prefill_nll": sum(prefill) / len(prefill),
                "decode_token_nll": decode,
                "prefill_token_nll": prefill,
                "first_raw_logprob_delta": abs(decode[0] - prefill[0]),
                "raw_logprobs_nonzero": True,
                "input_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }
            if reference is not None:
                loss = (
                    -reference[args.prefix - 1 : -1]
                    .float()
                    .log_softmax(-1)
                    .gather(1, torch.tensor(targets)[:, None])
                    .squeeze(1)
                )
                row["reference_nll"] = loss.mean().item()
                row["reference_token_nll"] = loss.tolist()
            receipt["cases"].append(row)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt["status"] = "complete"
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
