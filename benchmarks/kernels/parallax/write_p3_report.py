# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render the P3 report from measured JSON receipts without estimated rates."""

import json
from pathlib import Path

base = Path("/root/vllm_eda/receipts-p3")
data = json.loads((base / "p3-summary.json").read_text())
lines = []


def emit(text=""):
    lines.append(text)


def table(headers, rows):
    emit("| " + " | ".join(h.replace("|", "\\|") for h in headers) + " |")
    emit("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        emit("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
    emit()


def row(receipt, prompt, batch):
    return next(
        r
        for r in data["rows"]
        if r["receipt"] == receipt and r["prompt"] == prompt and r["batch"] == batch
    )


def paired(prefix, cases, decode=True):
    values = []
    for prompt, batch in cases:
        a = row(prefix + "-fp32.json", prompt, batch)
        b = row(prefix + "-fp16.json", prompt, batch)
        item = [prompt, batch, f"{a['rate']:,.0f}", f"{b['rate']:,.0f}"]
        if decode:
            item += [f"{a['mean_itl_ms']:.3f}", f"{b['mean_itl_ms']:.3f}"]
        values.append(item)
    return values


emit("# Lambda P3 serving quality, throughput and optimization")
emit()
emit(
    "All reported rates, losses and timings are **MEASURED**. Derived medians, "
    "differences and percentages use the linked measured trials. No estimated "
    "throughput, TP>1 result, commit or push is claimed."
)
emit()
emit(
    "Default Lambda parity follows the lead acceptance of P2b as within Mesh execution "
    "noise. P3 evaluates the explicit teacher/NLL/self optimization gates and retains "
    "the statistically underpowered greedy counts separately. Both opt-ins report "
    "4/8 large-margin first divergences, exceeding the observed maximum floor of "
    "3/8; that stricter count-envelope result is false in both verdict JSONs. "
    "Neither opt-in is promoted to the default; their combination is untested."
)
emit()
emit("## Settings and artifacts")
emit()
emit(
    "RTX 5090, SM120, one physical GPU per process, TP=1. Fork editable install: "
    "Torch 2.13.0+cu132, Triton 3.7.1. Lambda export revision "
    "`1b992902aa52ca487eee21160b45d66791d1e7dc`, step 21067; Kappa revision "
    "`15975e88320522284396bfe15c02e533d8b49dc4`, step 37898. Dense weights/KV and "
    "Lambda head BF16; recurrent cache FP32 or FP16 as labeled. All seven P2 "
    "Lambda arithmetic diagnostic switches remain false. Bounded logit scale, "
    "tied embeddings, unit-gain final norm, dense MSA and 2048 SWA stay enabled."
)
emit()
emit(
    "Serving: context reservation 8192, GPU-memory utilization 0.7, max batched "
    "tokens 32768, synchronized arrivals, temperature 0.8, top-p 0.95, "
    "FULL_DECODE_ONLY graphs with captures 1/4/256, decode-slot cache and in-place "
    "sampler. Decode uses 128 outputs, five trials after four warmups; decode "
    "rate counts 127 cached predictions per request and excludes prefill. "
    "Prefill uses one output, three trials after one warmup; rates include "
    "first output and scheduler time. All headline trials report zero preemptions."
)
emit()
emit(
    "GPU assignment: Lambda FP32=0, FP16=1, BF16-expert control=6; Kappa=7, "
    "on a separate GPU from Lambda. NLL original=2/3, fused=4/5. Baseline "
    "profiling=5 decode/4 prefill; fused profile=4; Kappa/tile profiles=7. "
    "DP8 uses GPUs 0–7, with all other GPU jobs finished before its launch. "
    "Diagnostic eager profiles and CUPTI sweeps are separate from headline trials."
)
emit()
emit(
    "The sparse-FP4 profile is mixed: batch 1/4 decode uses packed BF16 experts; "
    "batch 256 uses sparse_ternary_fp4, one activation component, ungated reduction. "
    "Profiler receipts directly show `dual_kernel<true,16,...>` and "
    "`dual_kernel<false,16,...>` on batch-256 decode. Prefill selects FP4 at "
    "256 or more scheduled tokens and block-M 64 at 2048 or more. The BF16 "
    "control sets single_sparse_fp4_components=0. State overrides are explicit "
    "in each benchmark receipt; older `model_config` fields preserve the source "
    "profile file (FP32), while `state_storage` records the actual engine override."
)
emit()
emit("## Lambda default decode")
emit()
table(
    ["Prompt", "Batch", "FP32 tok/s", "FP16 tok/s", "FP32 ITL ms", "FP16 ITL ms"],
    paired("decode-before", [(p, b) for p in (32, 128) for b in (1, 4, 256)]),
)
emit(
    "Receipts: [FP32](decode-before-fp32.json), [FP16](decode-before-fp16.json). "
    "ITL is the median per-request mean inter-token latency, then the median "
    "across trials. No tokenizer is used: the benchmark supplies token IDs."
)
emit()
control = [row("decode-bf16-control.json", p, 256) for p in (32, 128)]
table(
    ["BF16-expert control, FP32 state", "Prompt", "Batch", "tok/s"],
    [["GPU 6", r["prompt"], 256, f"{r['rate']:,.0f}"] for r in control],
)
emit("Receipt: [control](decode-bf16-control.json).")
emit()
emit("## Matched Lambda prefill")
emit()
table(
    ["Prompt", "Batch", "FP32 input tok/s", "FP16 input tok/s"],
    paired("prefill-matched", [(p, b) for p in (128, 2048) for b in (1, 16)], False),
)
emit(
    "Receipts: [FP32](prefill-matched-fp32.json), [FP16](prefill-matched-fp16.json). "
    "These exactly match native-integration-prefill.json shapes. The first "
    "exploratory prefill-before receipts used 512/2048 and batches 1/4/16; "
    "they are retained and are not substituted for matched results."
)
emit()
emit("## Simultaneous DP8, original default profile")
emit()
table(
    ["State", "Prompt", "Cohort tok/s", "Sum of per-GPU medians tok/s"],
    [
        [
            r["state"],
            r["prompt"],
            f"{r['cohort_decode_rate_median']:,.0f}",
            f"{r['sum_per_gpu_medians']:,.0f}",
        ]
        for r in data["dp8"]
    ],
)
table(
    ["GPU", "FP32 prompt32", "FP32 prompt128", "FP16 prompt32", "FP16 prompt128"],
    [[gpu] + [f"{r['per_gpu'][gpu]:,.0f}" for r in data["dp8"]] for gpu in range(8)],
)
emit(
    "Rates above are tok/s. Each trial uses a filesystem arrival barrier; "
    "cohort rate is 8×256×127 divided by latest last-token timestamp minus "
    "earliest first-token timestamp across the eight processes. Both aggregate "
    "definitions are measured, not an eight-times single-GPU estimate. "
    "Per-GPU receipts are dp8-{fp32,fp16}-gpu{0..7}.json; aggregates and trial "
    "values are in [p3-summary.json](p3-summary.json). No optimized DP8 or TP>1 "
    "throughput is claimed."
)
emit()
emit("## Same-box Kappa")
emit()
table(
    ["Prompt", "Batch", "FP32 tok/s", "FP16 tok/s", "FP32 ITL ms", "FP16 ITL ms"],
    paired("kappa-sameBox-decode", [(p, b) for p in (32, 128) for b in (1, 4, 256)]),
)
table(
    ["Prompt", "Batch", "FP32 input tok/s", "FP16 input tok/s"],
    paired(
        "kappa-sameBox-prefill", [(p, b) for p in (128, 2048) for b in (1, 16)], False
    ),
)
emit(
    "Receipts: kappa-sameBox-{decode,prefill}-{fp32,fp16}.json. Release profile "
    "uses FP32 GDN2 state; FP16 is an additional measured throughput control, "
    "not a new Kappa quality qualification. The owner’s roughly 28k decode "
    "report is reproduced by this workload’s FP16 results. This pinned snapshot "
    "has no LATEST.json at any path; the unique explicit step37898 export is "
    "used, documented in kappa-download.json. Only its 33 files were needed. "
    "The converter printed PAYLOADS_VERIFIED before conversion/profile creation."
)
emit()
emit("## Optimization deltas and quality screens")
emit()
emit(
    "Throughput receipts: [fused FP32 decode](decode-fused-fp32.json), "
    "[fused FP16 decode](decode-fused-fp16.json), "
    "[fused FP32 prefill](prefill-fused-fp32.json), "
    "[fused FP16 prefill](prefill-fused-fp16.json), "
    "[tiles FP32 decode](decode-tuned-fp32.json), "
    "[tiles FP16 decode](decode-tuned-fp16.json), "
    "[tiles FP32 prefill](prefill-tuned-fp32.json), "
    "[tiles FP16 prefill](prefill-tuned-fp16.json)."
)
emit()
values = []
for label in ("fused", "tuned"):
    for state in ("fp32", "fp16"):
        for prompt in (32, 128):
            before = row(f"decode-before-{state}.json", prompt, 256)["rate"]
            after = row(f"decode-{label}-{state}.json", prompt, 256)["rate"]
            values.append(
                [
                    label,
                    state,
                    prompt,
                    f"{before:,.0f}",
                    f"{after:,.0f}",
                    f"{100 * (after / before - 1):+.2f}%",
                ]
            )
table(
    ["Opt-in", "State", "Prompt / batch256", "Before tok/s", "After tok/s", "Delta"],
    values,
)
values = []
for label in ("fused", "tuned"):
    for state in ("fp32", "fp16"):
        before = row(f"prefill-matched-{state}.json", 2048, 16)["rate"]
        after = row(f"prefill-{label}-{state}.json", 2048, 16)["rate"]
        values.append(
            [
                label,
                state,
                f"{before:,.0f}",
                f"{after:,.0f}",
                f"{100 * (after / before - 1):+.2f}%",
            ]
        )
table(
    ["Opt-in / 16×2048", "State", "Before input tok/s", "After input tok/s", "Delta"],
    values,
)
emit(
    "Fused: one Triton EDA preparation kernel and one output norm/gate kernel, "
    "FP32 intermediates, explicit BF16 stores, FP fusion disabled. Tiles: "
    "BV64/four warps for more than four actual decode tokens, original BV16 "
    "for 1–4; prefill arithmetic is unchanged, so its small measured deltas "
    "are not attributed to a prefill optimization. Shared-input projections "
    "remain separate; no combined fusion+tile result is inferred."
)
emit()
values = []
for label in ("fused", "tuned"):
    d = json.loads((base / f"parity-{label}-verdict.json").read_text())
    s = d["summary"]
    shadow = json.loads((base / f"repro-{label}-shadow.json").read_text())
    values.append(
        [
            label,
            f"{100 * s['top1_agreement']:.4f}%",
            f"{s['mean_abs_dlogprob']:.6f}",
            f"{d['short_nll_delta']:+.6f}",
            "/".join(f"{r['delta']:+.6f}" for r in d["long_nll"]),
            f"{d['self_max']:.6f}",
            s["divergences_with_margin_ge_0_1"],
            d["requested_teacher_nll_self_screen_pass"],
            shadow["tokens_identical"] and shadow["logprobs_exact"],
        ]
    )
table(
    [
        "Opt-in",
        "Top-1",
        "Mean |dlp|",
        "Short NLL delta",
        "Long NLL deltas",
        "Self max",
        "Greedy ≥0.1 /8",
        "Teacher/NLL/self pass",
        "Shadow exact",
    ],
    values,
)
emit(
    "Receipts: parity-{fused,tuned}-verdict.json and repro-{fused,tuned}-shadow.json. "
    "Limits: top-1 ≥94.2264%, mean |dlp| ≤0.084353, absolute short/each-long "
    "NLL delta ≤0.01 nats, self max ≤0.578507. Greedy exact matches are 1/8 "
    "for each candidate; large-margin counts are 4/8, versus max(F2,F3,F5)=3/8. "
    "The final tile branch retains the old 1–4-token tile after the microbenchmark; "
    "the parity screen’s prefill and batch-eight continuations use the same "
    "code paths as its prior all-batch BV64 run. Final graph shadow/throughput "
    "receipts use the retained-small-batch branch."
)
emit()
emit("## Forced cached-decode NLL")
emit()
values = []
for label in ("", "fused-"):
    d = json.loads((base / f"decode-nll-{label}verdict.json").read_text())
    for group in ("windows", "long", "all"):
        g = d[group]
        values.append(
            [
                label.rstrip("-") or "default",
                group,
                g["tokens"],
                f"{g['reference_nll']:.6f}",
                f"{g['prefill_fp32_nll']:.6f}",
                f"{g['decode_fp32_nll']:.6f}",
                f"{g['decode_fp16_nll']:.6f}",
                f"{g['fp16_degradation']:+.6f}",
            ]
        )
table(
    [
        "Path",
        "Corpus",
        "Cached positions",
        "R_bf16 NLL",
        "Prefill NLL",
        "Decode FP32 NLL",
        "Decode FP16 NLL",
        "FP16 degradation",
    ],
    values,
)
for label in ("", "fused-"):
    d = json.loads((base / f"decode-nll-{label}verdict.json").read_text())
    emit(
        f"{label.rstrip('-') or 'Default'}: FP16 qualified on all cases = "
        f"**{d['fp16_qualified_all_cases']}**; maximum per-case degradation "
        f"**{max(r['fp16_degradation'] for r in d['cases']):+.6f} nats**. "
        f"Receipt: [decode-nll-{label}verdict.json](decode-nll-{label}verdict.json)."
    )
    emit()
emit(
    "Each sequence pre-fills 64 tokens, then forces its ground-truth continuation. "
    "The first prediction is made by prefill and excluded; 8 concatenated "
    "512-token sequences supply 3576 cached predictions, and the 2048/3072 "
    "real-text sequences supply 4990. The processor verifies every forced ID, "
    "and returned forced-token losses are nonzero. V2 sample processing copies "
    "logits before forcing; raw_logprobs scores the retained original logits, "
    "rather than the forced one-token distribution. No in-place sampler is "
    "used for NLL. Original and fused FP16 losses are compared with their "
    "matched FP32 paths, full-prefill and sealed BF16 mesh logits at identical "
    "positions. The one-request decode expert path is packed BF16; full-prefill "
    "uses mixed-FP4. This qualifies recurrent storage drift in the measured "
    "configuration, not batch-256 FP4 expert language quality. Tile-mode "
    "one-request NLL uses the unchanged BV16 branch."
)
emit()
emit(
    "Concatenated reference logits use sealed build_model(export, "
    "model_config=export/model_config.json, expert_format=bf16, "
    "eda_kernel_backend=flashqla_eda), calling the bench-lane direct export "
    "loader. Reference artifacts remain under /root/vllm_eda/reference-decode; "
    "their receipt and hashes are preserved. Long R_bf16 artifacts are the "
    "same P2b files. The exact token corpus and per-case input hashes are retained."
)
emit()
emit("## Profiles, validation and limitations")
emit()
emit(
    "Profile attribution is in profile-*-kernels.json (sum of GPU kernel "
    "durations; not generation wall time) and profile-{before-decode, "
    "before-prefill,fused,tuned,kappa}.json (eager module events including "
    "instrumentation and host-launch gaps). Recurrent, MoE, attention, projection "
    "and logits/sampler timings are separate. EDA has twelve linear calls per "
    "layer before either opt-in; Kappa uses joined GDN2 projections and its "
    "decode_gated recurrence. The component table below uses module totals "
    "from the last matching eager step and is diagnostic, not a headline rate."
)
emit()
values = []
for name in (
    "profile-before-decode",
    "profile-before-prefill",
    "profile-fused",
    "profile-tuned",
    "profile-kappa",
):
    d = json.loads((base / (name + ".json")).read_text())
    for r in d["rows"]:
        times = r["component_timing_trials"][0][0]["last_step_gpu_ms"]

        def total(needle, times=times):
            return sum(
                v for k, v in times.items() if k.startswith("layer.") and needle in k
            )

        projections = sum(v for k, v in times.items() if k.startswith("mixer."))
        values.append(
            [
                name,
                r["prompt_tokens"],
                r["concurrency"],
                f"{total('EDALayer') + total('GDN2Layer'):.3f}",
                f"{projections:.3f}",
                f"{total('ExpertLayer'):.3f}",
                f"{total('AttentionLayer'):.3f}",
                f"{times.get('model', 0):.3f}",
            ]
        )
table(
    [
        "Profile",
        "Prompt",
        "Batch",
        "Mixers ms",
        "Mixer linears ms (nested)",
        "MoE ms",
        "Attention ms",
        "Model ms",
    ],
    values,
)
emit(
    "The original decode _scan consumed 499.308 ms over 2286 calls; "
    "same-box Kappa _decode_gated consumed 424.924 ms over 2286 calls "
    "in their respective profiled generations. See profile-before-kernels.json "
    "and profile-kappa-kernels.json. Whole-model Torch elementwise/copy families "
    "and narrow projection calls add overhead; their totals are not all assigned "
    "to EDA. Profiles contain prefill and 127 decode steps. The profiler reuses "
    "its first filename prefix on subsequent cases; trace order and receipt "
    "workloads, not that stale filename prefix, identify decode versus prefill."
)
emit()
emit(
    "CUPTI cold-L2 scan sweep: [profile-scan-sweep.json](profile-scan-sweep.json), "
    "48 configurations at batch 1/4/256 and FP32/FP16, BV16/32/64/128, "
    "warps4/8; output and cached state checked before timing. At batch256 "
    "BV16/4 → BV64/4: FP32 236.831 →210.623 µs; FP16 141.3115 →107.136 µs. "
    "Small batches retained the original tile. State-byte bandwidth is calculated "
    "as two times state bytes divided by measured time. The initial missing-CUPTI "
    "warm-cache fallback is explicitly discarded in "
    "profile-scan-sweep-warm-fallback.json."
)
emit()
emit(
    "Validation: **41 tests passed**: EDA 20, loader/profile 14, sampler 6, "
    "scheduler 1. Logs: p3-tests-{eda,loader,sampler,scheduler}.log. Ruff 0.14.0 "
    "check and format checks are in ruff-p3.log. Shell runners pass bash -n. "
    "Executed model/kernel source hashes are in each serving/parity receipt; "
    "P3_SOURCE_SHA256.json captures the final source. P3_COMMANDS.sh and the "
    "run_p3_*.sh scripts preserve complete commands/settings. No combined "
    "optimization, broad language evaluation or optimized DP8 rate is claimed."
)
emit()
emit(
    "Boundary incident: one loader test was invoked outside the isolated runner "
    "and created temporary fixtures under remote /tmp/pytest-of-root. Subsequent "
    "tests were corrected to /root/vllm_eda/tmp and scoped cache paths. No sealed "
    "reference source/environment or main checkout was modified. No commits or "
    "pushes were made. Public downloads used token=False; no secrets were copied."
)
emit()
emit(
    "Open issues: the 4/8 greedy counts for both opt-ins exceed the observed "
    "3/8 envelope and are statistically underpowered; neither becomes the "
    "default. Actual all-batch FP4 teacher parity remains the P2b issue. "
    "TP>1, combined pointwise+tiles, optimized DP8 and general language quality "
    "are not qualified. Separate EDA projections remain an optimization "
    "opportunity; no unmeasured joined-GEMM speedup is asserted."
)
(base / "P3_REPORT.md").write_text("\n".join(lines) + "\n")
