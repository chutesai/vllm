# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline vLLM serving benchmark; real scheduler, sampling and KV lifecycle."""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

from vllm import LLM, SamplingParams

_LIVE_LLM = None


def percentile95(values):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def main():
    global _LIVE_LLM
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--prompts", default="512,2048")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--outputs", help="Comma-separated generated lengths")
    parser.add_argument("--context-limit", type=int)
    parser.add_argument("--engine-context", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--full-decode-graph", action="store_true")
    parser.add_argument("--profile-dir")
    parser.add_argument("--capture-sizes")
    parser.add_argument("--synchronize-arrivals", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--cases-file", help="JSON list of explicit workload rows")
    parser.add_argument("--auto-context", action="store_true")
    parser.add_argument("--max-seqs", type=int)
    parser.add_argument("--component-timing", action="store_true")
    parser.add_argument("--inplace-sampler", action="store_true")
    parser.add_argument("--cpu-profile")
    parser.add_argument("--decode-slot-cache", action="store_true")
    parser.add_argument("--decode-slot-shadow", action="store_true")
    parser.add_argument("--sample-logprobs", type=int)
    args = parser.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    prompts = [int(x) for x in args.prompts.split(",")]
    outputs_to_test = (
        [int(x) for x in args.outputs.split(",")] if args.outputs else [args.tokens]
    )
    cases = (
        json.loads(Path(args.cases_file).read_text())
        if args.cases_file
        else [
            {"prompt_tokens": p, "output_tokens": o, "concurrency": b}
            for o in outputs_to_test
            for p in prompts
            for b in batches
        ]
    )
    if not cases or any(
        not isinstance(c[k], int) or c[k] < 1
        for c in cases
        for k in ("prompt_tokens", "output_tokens", "concurrency")
    ):
        raise ValueError("Cases must contain positive integer lengths and concurrency")
    batches = [c["concurrency"] for c in cases]
    config_data = json.loads((Path(args.model) / "config.json").read_text())
    context_limit = args.context_limit or config_data["max_position_embeddings"]
    workload_context = max(c["prompt_tokens"] + c["output_tokens"] for c in cases)
    if not args.auto_context and workload_context > context_limit:
        raise ValueError("Input plus output exceeds supported context")
    engine_context = (
        -1 if args.auto_context else (args.engine_context or workload_context)
    )
    if not args.auto_context and (
        engine_context > context_limit or engine_context < workload_context
    ):
        raise ValueError("Engine context must cover workload and respect model limit")
    extra = {}
    if args.decode_slot_cache:
        if args.component_timing:
            raise ValueError(
                "Decode-slot caching and diagnostics are separate experiments"
            )
        extra["worker_extension_cls"] = (
            "vllm.v1.worker.gpu.parallax_sampler.PipelineOptimizationWorker"
            if args.inplace_sampler
            else "vllm.v1.core.sched.parallax_decode.DecodeSlotCacheWorker"
        )
        extra["scheduler_cls"] = (
            "vllm.v1.core.sched.parallax_decode.ShadowDecodeSlotScheduler"
            if args.decode_slot_shadow
            else "vllm.v1.core.sched.parallax_decode.DecodeSlotScheduler"
        )
    elif args.decode_slot_shadow:
        raise ValueError("Shadow mode requires --decode-slot-cache")
    if args.cpu_profile and not args.component_timing:
        raise ValueError("CPU profiling uses the diagnostic worker extension")
    if args.inplace_sampler:
        if args.component_timing:
            raise ValueError(
                "Sampler optimization and diagnostic wrapper are separate tests"
            )
        if not args.decode_slot_cache:
            extra["worker_extension_cls"] = (
                "vllm.v1.worker.gpu.parallax_sampler.InplaceSamplerWorker"
            )
    if args.component_timing:
        if not config_data.get("component_timing", False):
            raise ValueError("Component timing requires model config opt-in")
        extra["worker_extension_cls"] = (
            "vllm.v1.worker.gpu.parallax_timing.ComponentTimingWorker"
        )
    if args.full_decode_graph:
        extra["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    if args.capture_sizes:
        if not args.full_decode_graph:
            raise ValueError("Capture sizes require --full-decode-graph")
        extra["compilation_config"]["cudagraph_capture_sizes"] = [
            int(x) for x in args.capture_sizes.split(",")
        ]
    if args.profile_dir:
        extra["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": args.profile_dir,
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
        }
    progress_path = args.output + ".progress.json"
    write_json(progress_path, {"phase": "initializing", "time": time.time()})
    llm = LLM(
        model=args.model,
        load_format=args.load_format,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=engine_context,
        max_num_seqs=args.max_seqs or max(batches),
        max_num_batched_tokens=args.max_batched_tokens,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.eager,
        seed=20260909,
        disable_log_stats=False,
        **extra,
    )
    _LIVE_LLM = llm
    if args.decode_slot_cache:
        decode_slot_install = llm.collective_rpc(
            "parallax_set_decode_slot_cache", args=(True, args.decode_slot_shadow)
        )
    if args.inplace_sampler:
        llm.collective_rpc("parallax_set_inplace_sampler", args=(True,))
    if args.component_timing:
        llm.collective_rpc("parallax_install_sampler_timing")
    engine_context = llm.llm_engine.vllm_config.model_config.max_model_len
    peaks = {}
    manager = llm.llm_engine.logger_manager
    original_record = manager.record

    def record(scheduler_stats, iteration_stats, *positional, **keywords):
        if scheduler_stats is not None:
            for name, value in (
                ("peak_kv_cache_fraction", scheduler_stats.kv_cache_usage),
                ("peak_running_requests", scheduler_stats.num_running_reqs),
                ("peak_waiting_requests", scheduler_stats.num_waiting_reqs),
            ):
                peaks[name] = max(peaks.get(name, 0), value)
        if iteration_stats is not None:
            peaks["preemptions"] = (
                peaks.get("preemptions", 0) + iteration_stats.num_preempted_reqs
            )
        return original_record(
            scheduler_stats, iteration_stats, *positional, **keywords
        )

    manager.record = record
    result = {
        "model": args.model,
        "weights": (args.load_format + " seeded initialization")
        if args.load_format in ("dummy", "parallax_random")
        else "loaded model weights (" + args.load_format + ")",
        "dtype": "bfloat16",
        "sampling": {"temperature": args.temperature, "top_p": args.top_p},
        "sample_logprobs": args.sample_logprobs,
        "capture_sizes_requested": args.capture_sizes,
        "synchronized_arrivals": args.synchronize_arrivals,
        "eager": args.eager,
        "full_decode_graph": args.full_decode_graph,
        "max_batched_tokens": args.max_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "timing": "generation wall time including prefill and scheduler",
        "output_lengths": outputs_to_test,
        "context_limit": context_limit,
        "engine_context": engine_context,
        "warmup_generations_per_shape": args.warmups,
        "repeats": args.repeats,
        "inplace_sampler": args.inplace_sampler,
        "decode_slot_cache": args.decode_slot_cache,
        "decode_slot_shadow": args.decode_slot_shadow,
        "decode_slot_cache_install": decode_slot_install
        if args.decode_slot_cache
        else None,
        "rows": [],
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    import vllm
    import vllm.model_executor.models.parallax as parallax_model

    source = Path(vllm.__file__).parent
    implementation_files = {Path(parallax_model.__file__)}
    for pattern in (
        "model_executor/layers/fused_moe/parallax/**/*.py",
        "model_executor/layers/attention/ops/parallax/**/*.py",
        "model_executor/layers/attention/block*residual*.py",
        "model_executor/layers/mamba/gdn/parallax*.py",
        "model_executor/layers/parallax_linear.py",
        "model_executor/model_loader/parallax*.py",
        "transformers_utils/configs/parallax*",
        "v1/core/sched/parallax*.py",
        "v1/worker/gpu/parallax*.py",
    ):
        implementation_files.update(source.glob(pattern))
    result["vllm"] = vllm.__version__
    result["implementation_sha256"] = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(implementation_files)
        if p.suffix in (".py", ".cu", ".cuh", ".json", ".toml")
    }
    identity = Path(args.model) / "artifact_identity.json"
    if identity.exists():
        result["artifact_identity"] = json.loads(identity.read_text())
    config = Path(args.model) / "config.json"
    if config.exists():
        result["model_config"] = json.loads(config.read_text())
    write_json(args.output, result)

    def generate(inputs, sampling):
        if not args.synchronize_arrivals:
            return llm.generate(inputs, sampling, use_tqdm=False)
        core = llm.llm_engine.engine_core
        core.call_utility("pause_scheduler", "keep", False)
        try:
            llm.enqueue(inputs, sampling, use_tqdm=False)
        finally:
            core.call_utility("resume_scheduler")
        return llm.wait_for_completion(use_tqdm=False)

    for case in cases:
        length, generated, batch = (
            case[k] for k in ("prompt_tokens", "output_tokens", "concurrency")
        )
        if length + generated > min(context_limit, engine_context):
            status = (
                "unsupported_context"
                if length + generated > context_limit
                else "exceeds_engine_capacity"
            )
            result["rows"].append({**case, "status": status})
            write_json(args.output, result)
            continue
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=generated,
            min_tokens=generated,
            ignore_eos=True,
            detokenize=False,
            logprobs=args.sample_logprobs,
        )
        inputs = [
            {
                "prompt_token_ids": [
                    2 + ((i * 37 + j * 71) % 30000) for i in range(length)
                ]
            }
            for j in range(batch)
        ]
        write_json(
            progress_path, {"phase": "running", "case": case, "time": time.time()}
        )
        for _ in range(args.warmups):
            generate(inputs, sampling)
        samples = []
        metrics = []
        token_hashes = []
        logprob_hashes = []
        component_timings = []
        for trial_index in range(args.repeats):
            peaks.clear()
            if args.cpu_profile and trial_index == 0:
                llm.collective_rpc("parallax_start_cpu_profile")
            start = time.perf_counter()
            outputs = generate(inputs, sampling)
            elapsed = time.perf_counter() - start
            if args.cpu_profile and trial_index == 0:
                llm.collective_rpc(
                    "parallax_stop_cpu_profile", args=(args.cpu_profile,)
                )
            if args.component_timing:
                component_timings.append(
                    llm.collective_rpc("parallax_component_timing", args=(batch,))
                )
            token_hashes.append(
                hashlib.sha256(
                    json.dumps(
                        [list(o.outputs[0].token_ids) for o in outputs],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            )
            if args.sample_logprobs is not None:
                scored = [
                    [
                        [
                            (token, lp.logprob, lp.rank)
                            for token, lp in sorted(step.items())
                        ]
                        for step in output.outputs[0].logprobs
                    ]
                    for output in outputs
                ]
                logprob_hashes.append(
                    hashlib.sha256(json.dumps(scored).encode()).hexdigest()
                )
            count = sum(len(o.outputs[0].token_ids) for o in outputs)
            assert count == batch * generated
            stats = [o.metrics for o in outputs]
            assert all(s is not None and not s.is_corrupted for s in stats)
            decode_span = max(s.last_token_ts for s in stats) - min(
                s.first_token_ts for s in stats
            )
            metrics.append(
                {
                    **peaks,
                    "ttft_median_seconds": statistics.median(
                        s.first_token_latency for s in stats
                    ),
                    "ttft_max_seconds": max(s.first_token_latency for s in stats),
                    "ttft_min_seconds": min(s.first_token_latency for s in stats),
                    "ttft_p95_seconds": percentile95(
                        s.first_token_latency for s in stats
                    ),
                    "decode_output_tokens_per_second": batch
                    * (generated - 1)
                    / decode_span
                    if generated > 1 and decode_span > 0
                    else None,
                    "per_request_mean_itl_seconds": statistics.median(
                        (s.last_token_ts - s.first_token_ts) / (generated - 1)
                        for s in stats
                    )
                    if generated > 1
                    else None,
                    "per_request_mean_itl_p95_seconds": percentile95(
                        (s.last_token_ts - s.first_token_ts) / (generated - 1)
                        for s in stats
                    )
                    if generated > 1
                    else None,
                }
            )
            samples.append(elapsed)
        seconds = statistics.median(samples)
        row = {
            "status": "complete",
            "prompt_tokens": length,
            "output_tokens": generated,
            "concurrency": batch,
            "seconds": seconds,
            "trials_seconds": samples,
            "fresh_prefill_tokens_per_second_including_first_output": batch
            * length
            / seconds
            if generated == 1
            else None,
            "output_tokens_per_second_including_prefill": count / seconds,
            "input_plus_output_tokens_per_second": batch
            * (length + generated)
            / seconds,
            "request_timing_trials": metrics,
            "output_token_sha256_per_trial": token_hashes,
        }
        result["rows"].append(row)
        if args.component_timing:
            row["component_timing_trials"] = component_timings
        if args.inplace_sampler:
            row["inplace_sampler_report"] = llm.collective_rpc(
                "parallax_inplace_sampler_report"
            )
        if args.decode_slot_cache:
            row["decode_slot_cache_report"] = llm.collective_rpc(
                "parallax_decode_slot_cache_report"
            )
        if args.sample_logprobs is not None:
            row["logprob_sha256_per_trial"] = logprob_hashes
        write_json(args.output, result)
        print(json.dumps(row), flush=True)
        if args.profile_dir:
            llm.start_profile(f"p{length}_b{batch}")
            generate(inputs, sampling)
            llm.stop_profile()
    write_json(progress_path, {"phase": "complete", "time": time.time()})


if __name__ == "__main__":
    try:
        main()
    finally:
        if _LIVE_LLM is not None:
            _LIVE_LLM.llm_engine.engine_core.shutdown()
