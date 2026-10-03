# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Matched greedy inference screen; this is not a throughput or quality benchmark."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument(
        "--baseline", help="Saved tokens/logprobs from the reference runtime"
    )
    parser.add_argument(
        "--save-baseline", help="Save this run for a matched subsequent comparison"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--shadow", action="store_true")
    args = parser.parse_args()
    if args.shadow and not args.optimized:
        parser.error("--shadow requires --optimized")
    corpus = json.loads(Path(args.inputs).read_text())
    extra = {}
    if args.optimized:
        extra["worker_extension_cls"] = (
            "vllm.v1.worker.gpu.parallax_sampler.PipelineOptimizationWorker"
        )
        extra["scheduler_cls"] = (
            "vllm.v1.core.sched.parallax_decode.ShadowDecodeSlotScheduler"
            if args.shadow
            else "vllm.v1.core.sched.parallax_decode.DecodeSlotScheduler"
        )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
        enforce_eager=True,
        gpu_memory_utilization=0.7,
        **extra,
    )
    try:
        if args.optimized:
            llm.collective_rpc(
                "parallax_set_decode_slot_cache", args=(True, args.shadow)
            )
            llm.collective_rpc("parallax_set_inplace_sampler", args=(True,))
        llm.llm_engine.engine_core.call_utility("pause_scheduler", "keep", False)
        llm.enqueue(
            [
                {
                    "prompt_token_ids": corpus["prompts"][i % len(corpus["prompts"])][
                        :128
                    ]
                }
                for i in range(256)
            ],
            SamplingParams(
                temperature=0,
                max_tokens=128,
                ignore_eos=True,
                logprobs=0,
                detokenize=False,
            ),
            use_tqdm=False,
        )
        llm.llm_engine.engine_core.call_utility("resume_scheduler")
        outputs = llm.wait_for_completion(use_tqdm=False)
        tokens = [list(o.outputs[0].token_ids) for o in outputs]
        lp = torch.tensor(
            [
                [
                    step[token].logprob
                    for token, step in zip(
                        o.outputs[0].token_ids, o.outputs[0].logprobs, strict=True
                    )
                ]
                for o in outputs
            ],
            dtype=torch.float64,
        )
        if args.save_baseline:
            torch.save({"tokens": tokens, "lp": lp}, args.save_baseline)
        base = torch.load(args.baseline, weights_only=True) if args.baseline else None
        receipt = {
            "scope": (
                "256 sequences x 128 prompt + 128 generated; "
                "matched greedy numerical screen, not a quality evaluation"
            ),
            "tokens_identical": tokens == base["tokens"] if base else None,
            "logprobs_exact": torch.equal(lp, base["lp"]) if base else None,
            "max_abs_logprob_delta": float((lp - base["lp"]).abs().max())
            if base
            else None,
            "optimized": args.optimized,
            "shadow": args.shadow,
            "token_sha256": hashlib.sha256(
                json.dumps(tokens, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        import vllm.model_executor.models.parallax as parallax_model

        receipt["model_path"] = parallax_model.__file__
        import vllm

        receipt["vllm_version"] = vllm.__version__
        receipt["vllm_path"] = vllm.__file__
        if args.optimized:
            receipt["decode_slot_cache"] = llm.collective_rpc(
                "parallax_decode_slot_cache_report"
            )
            receipt["inplace_sampler"] = llm.collective_rpc(
                "parallax_inplace_sampler_report"
            )
        Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        if base is not None and (
            not receipt["tokens_identical"] or not receipt["logprobs_exact"]
        ):
            raise RuntimeError("Standalone port changed numerical outputs")
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
