# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mesh arithmetic, greedy divergence and fork prefill/decode parity receipts."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from lambda_parity_metrics import teacher_top1

from vllm import LLM, SamplingParams


def summarize(values):
    x = torch.tensor(values, dtype=torch.float64)
    return {"count": x.numel(), "max": float(x.max()), "mean": float(x.mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--references", nargs="+", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--state", choices=("fp32", "fp16"), default="fp32")
    p.add_argument("--limit", type=int)
    p.add_argument("--greedy-prompts", type=int, default=8)
    p.add_argument("--greedy-tokens", type=int, default=128)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--trace", type=Path)
    p.add_argument("--trace-inputs", type=Path)
    p.add_argument("--full-accum", action="store_true")
    p.add_argument("--overrides", type=json.loads, default={})
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--teacher-batch-size", type=int, default=1)
    p.add_argument("--greedy-inputs", type=Path)
    p.add_argument("--defer-greedy-reference", action="store_true")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "measurement_label": "MEASURED",
        "status": "running",
        "argv": sys.argv,
        "model": args.model,
        "state_storage": args.state,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "references": {},
        "greedy": {},
        "self_consistency": [],
        "teacher_artifacts": {},
        "nll": [],
        "overrides": args.overrides,
        "teacher_top1_method": "argmax_first_token_id",
        "greedy_inputs_sha256": hashlib.sha256(
            args.greedy_inputs.read_bytes()
        ).hexdigest()
        if args.greedy_inputs
        else None,
    }

    def save():
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")

    kwargs = {}
    if args.trace:
        kwargs["worker_extension_cls"] = (
            "vllm.v1.worker.gpu.parallax_trace.LambdaTraceWorker"
        )
    if args.graph:
        kwargs["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 4, 8],
        }
    overrides = {"gdn2_state_storage": args.state}
    overrides.update(args.overrides)
    if args.full_accum:
        overrides["bf16_full_accum"] = True
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=4096,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=max(4096, 128 * args.teacher_batch_size),
        enable_prefix_caching=False,
        enforce_eager=not args.graph,
        gpu_memory_utilization=0.85,
        hf_overrides=overrides,
        **kwargs,
    )
    receipt["effective_config"] = (
        llm.llm_engine.vllm_config.model_config.hf_config.to_dict()
    )
    root = Path(__file__).resolve().parents[3]
    receipt["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "vllm/model_executor/models/parallax.py",
            "vllm/model_executor/model_loader/parallax.py",
            "vllm/transformers_utils/configs/parallax.py",
            "vllm/model_executor/layers/mamba/eda/parallax_eda.py",
            "vllm/model_executor/layers/mamba/eda/parallax_eda_decode.py",
            "vllm/model_executor/layers/mamba/eda/parallax_eda_gated.py",
            "vllm/model_executor/layers/mamba/eda/parallax_eda_fused.py",
            "benchmarks/kernels/parallax/lambda_parity_metrics.py",
        )
    }
    save()
    if args.trace:
        llm.collective_rpc(
            "lambda_install_trace",
            args=(
                str(args.trace),
                str(args.trace_inputs) if args.trace_inputs else None,
            ),
        )
    try:
        names = sorted(x.stem for x in args.references[0].glob("prompt_*.pt"))
        if args.limit:
            names = names[: args.limit]
        names += sorted(x.stem for x in args.references[0].glob("long_*.pt"))
        for refdir in args.references:
            receipt["references"][refdir.name] = {"teacher": [], "greedy": []}
        teacher_outputs = {}
        for name in names:
            data = torch.load(args.references[0] / (name + ".pt"), weights_only=True)
            ids = data["input_ids"].tolist()
            if name not in teacher_outputs:
                group = names[
                    names.index(name) : names.index(name) + args.teacher_batch_size
                ]
                # Perturb short batch composition; keep long contexts independent.
                if name.startswith("long"):
                    group = [name]
                else:
                    group = [s for s in group if s.startswith("prompt")]
                group_ids = [
                    torch.load(args.references[0] / (s + ".pt"), weights_only=True)[
                        "input_ids"
                    ].tolist()
                    for s in group
                ]
                outputs = llm.generate(
                    [{"prompt_token_ids": x} for x in group_ids],
                    SamplingParams(
                        temperature=0,
                        max_tokens=1,
                        ignore_eos=True,
                        detokenize=False,
                        prompt_logprobs=20,
                    ),
                    use_tqdm=False,
                )
                teacher_outputs.update(zip(group, outputs, strict=True))
            output = teacher_outputs.pop(name)
            steps = output.prompt_logprobs[1:]
            receipt["teacher_artifacts"][name] = {
                "input_ids": ids,
                "top1": [
                    teacher_top1({t: v.logprob for t, v in step.items()})
                    for step in steps
                ],
                "logprobs": [
                    {str(t): v.logprob for t, v in step.items()} for step in steps
                ],
            }
            receipt["nll"].append(
                {
                    "case": name,
                    "tokens": len(steps),
                    "mean": -sum(
                        step[t].logprob for step, t in zip(steps, ids[1:], strict=True)
                    )
                    / len(steps),
                }
            )
            for refdir in args.references:
                ref = torch.load(refdir / (name + ".pt"), weights_only=True)
                logits = ref["logits"][:-1]
                top = logits.argmax(-1)
                chosen = logits.log_softmax(-1).gather(1, top[:, None]).squeeze(1)
                delta, agreement = [], 0
                for j, step in enumerate(output.prompt_logprobs[1:]):
                    token = int(top[j])
                    if token not in step:
                        raise RuntimeError(
                            f"Reference top1 absent from fork top20: {name}:{j}"
                        )
                    delta.append(abs(step[token].logprob - float(chosen[j])))
                    agreement += receipt["teacher_artifacts"][name]["top1"][j] == token
                row = {
                    "case": name,
                    "tokens": len(delta),
                    "top1_agreement": agreement / len(delta),
                    "chosen_logprob_delta": summarize(delta),
                    "max_abs_logit_diff": None,
                    "raw_logits_obtainable": False,
                }
                receipt["references"][refdir.name]["teacher"].append(row)
            save()
            if args.trace and name == "prompt_000":
                llm.collective_rpc("lambda_save_trace")
                trace = torch.load(args.trace, weights_only=True)
                if "logits" in trace:
                    raw = trace["logits"].float()
                    for refdir in args.references:
                        reference = torch.load(
                            refdir / (name + ".pt"), weights_only=True
                        )["logits"]
                        row = receipt["references"][refdir.name]["teacher"][-1]
                        count = min(raw.shape[0], reference.shape[0])
                        row.update(
                            raw_logits_obtainable=True,
                            raw_logit_positions=count,
                            max_abs_logit_diff=float(
                                (raw[:count] - reference[:count]).abs().max()
                            ),
                        )
                    save()
            print(name, flush=True)
        prompts = (
            json.loads(args.greedy_inputs.read_text())["prompts"][: args.greedy_prompts]
            if args.greedy_inputs
            else [
                torch.load(
                    args.references[0] / f"greedy_{i:03d}.pt", weights_only=True
                )["prompt_ids"].tolist()
                for i in range(args.greedy_prompts)
            ]
        )
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in prompts],
            SamplingParams(
                temperature=0,
                max_tokens=args.greedy_tokens,
                ignore_eos=True,
                logprobs=2,
                detokenize=False,
            ),
            use_tqdm=False,
        )
        for i, (ids, output) in enumerate(zip(prompts, outputs, strict=True)):
            generated = list(output.outputs[0].token_ids)
            lp = [
                s[t].logprob
                for t, s in zip(generated, output.outputs[0].logprobs, strict=True)
            ]
            for refdir in args.references:
                if args.defer_greedy_reference:
                    continue
                ref = torch.load(refdir / f"greedy_{i:03d}.pt", weights_only=True)
                expected = ref["tokens"].tolist()[: args.greedy_tokens]
                first = next(
                    (
                        j
                        for j, (a, b) in enumerate(
                            zip(generated, expected, strict=True)
                        )
                        if a != b
                    ),
                    None,
                )
                margin = (
                    None
                    if first is None
                    else float(
                        ref["top20_logprobs"][first, 0]
                        - ref["top20_logprobs"][first, 1]
                    )
                )
                receipt["references"][refdir.name]["greedy"].append(
                    {
                        "sequence": i,
                        "exact_match": first is None,
                        "first_divergence": first,
                        "reference_top2_margin": margin,
                    }
                )
            teacher = llm.generate(
                [{"prompt_token_ids": ids + generated}],
                SamplingParams(
                    temperature=0,
                    max_tokens=1,
                    ignore_eos=True,
                    prompt_logprobs=0,
                    detokenize=False,
                ),
                use_tqdm=False,
            )[0]
            delta = [
                abs(a - teacher.prompt_logprobs[len(ids) + j][t].logprob)
                for j, (a, t) in enumerate(zip(lp, generated, strict=True))
            ]
            receipt["self_consistency"].append(
                {"sequence": i, "delta": summarize(delta)}
            )
            receipt["greedy"][str(i)] = {
                "tokens": generated,
                "logprobs": lp,
                "top2_margins": [
                    sorted((v.logprob for v in step.values()), reverse=True)[0]
                    - sorted((v.logprob for v in step.values()), reverse=True)[1]
                    for step in output.outputs[0].logprobs
                ],
            }
            save()
        receipt["status"] = "complete"
        for result in receipt["references"].values():
            short = [r for r in result["teacher"] if r["case"].startswith("prompt")]
            total = sum(r["tokens"] for r in short)
            result["summary"] = {
                "top1_agreement": sum(r["top1_agreement"] * r["tokens"] for r in short)
                / total,
                "mean_abs_dlogprob": sum(
                    r["chosen_logprob_delta"]["mean"] * r["tokens"] for r in short
                )
                / total,
                "max_abs_dlogprob": max(
                    r["chosen_logprob_delta"]["max"] for r in short
                ),
                "greedy_exact_matches": sum(r["exact_match"] for r in result["greedy"]),
                "divergences_with_margin_ge_0_1": sum(
                    r["reference_top2_margin"] is not None
                    and r["reference_top2_margin"] >= 0.1
                    for r in result["greedy"]
                ),
            }
        receipt["self_consistency_max"] = max(
            r["delta"]["max"] for r in receipt["self_consistency"]
        )
        save()
    except Exception as exc:
        receipt.update(status="failed", error=repr(exc))
        save()
        raise
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
