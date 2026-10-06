# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write a receipt-backed P4 report; refuse incomplete final qualification."""

import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--selected", required=True)
a = p.parse_args()
base = a.root


def read(name):
    return json.loads((base / name).read_text())


s = read("p4-summary.json")
g = read("p4-greedy32-verdict.json")
nll = read("p4-decode-nll-final-verdict.json")
assert g["candidates"][a.selected]["within_max_floor_count"]
assert read(f"p4-{a.selected}-screen.json")["requested_teacher_nll_self_screen_pass"]
assert nll["fp16_qualified_all_cases"]
assert len(s["dp8"]) == 4
for dtype in ("fp32", "fp16"):
    shadow = read(f"p4-final-{dtype}-shadow.json")
    assert shadow["tokens_identical"] and shadow["logprobs_exact"]

lines = []


def emit(text=""):
    lines.append(text)


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", r"\|")

    emit("| " + " | ".join(map(cell, headers)) + " |")
    emit("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        emit("| " + " | ".join(map(cell, row)) + " |")
    emit()


def serving(name, prompt, batch):
    return next(
        r
        for r in s["rows"]
        if r["receipt"] == name and r["prompt"] == prompt and r["batch"] == batch
    )


def preserved_rate(name, prompt, batch):
    r = next(
        r
        for r in read(name)["rows"]
        if r["prompt_tokens"] == prompt and r["concurrency"] == batch
    )
    assert r["status"] == "complete"
    if r["output_tokens"] > 1:
        return statistics.median(
            t["decode_output_tokens_per_second"] for t in r["request_timing_trials"]
        )
    return r["fresh_prefill_tokens_per_second_including_first_output"]


emit("# Lambda P4 EDA optimization and qualification")
emit()
emit(
    "All throughput, loss, divergence counts and timings in this report "
    "are **MEASURED**, or differences derived from linked measured "
    "receipts. State-byte bandwidth in the kernel sweep uses **ESTIMATED** "
    "read/write byte counts. No estimated serving rate is used."
)
emit()
emit(
    f"Final Lambda profile: **{a.selected}**. The profile generator selects "
    f"this qualified combination for Lambda; `--eda-optimization none` "
    f"reproduces the previous configuration. Kappa profile generation and "
    f"model arithmetic are unchanged. No commits or pushes were made."
)
emit()
emit("## Settings")
emit()
emit(
    "RTX 5090 SM120; TP=1 per process. Fork Torch 2.13.0+cu132 / Triton "
    "3.7.1; sealed mesh oracle Torch 2.12.1+cu130. Export "
    "`chutesai/parallax-8b-lambda` revision "
    "`1b992902aa52ca487eee21160b45d66791d1e7dc`, step 21067. BF16 dense "
    "weights/KV/head, explicit FP32 or FP16 K-major recurrent state. Mixed "
    "experts: packed BF16 below 256 scheduled tokens; sparse ternary FP4 "
    "at/above 256. Shared expert, tied head, bounded learned logit scale, "
    "unit-gain final norm, dense MSA and 2048 SWA policies remain active."
)
emit()
emit(
    "Headline serving: context reservation 8192, max batched tokens 32768, "
    "GPU-memory utilization 0.7, synchronized arrivals, temperature 0.8 / "
    "top-p 0.95, FULL_DECODE_ONLY CUDA graphs captured at 1/4/256, "
    "decode-slot cache and in-place sampler. Decode: 128 outputs, five "
    "trials after four warmups; rates count 127 cached predictions/request "
    "and exclude prefill. Prefill: one output, three trials after one "
    "warmup; input rates include first output and scheduler time. The "
    "complete argv, effective config, GPU assignment and implementation "
    "hashes are in each receipt. Diagnostic eager component profiles "
    "include instrumentation overhead and are separate from headline "
    "serving."
)
emit()
emit("## Step measurements and gates")
emit()
rows = []
prior = {
    (dtype, prompt): preserved_rate(f"decode-before-{dtype}.json", prompt, 256)
    for dtype in ("fp32", "fp16")
    for prompt in (32, 128)
}
for step in ("combined", "joined", "gated", "tuned", "stable"):
    for dtype in ("fp32", "fp16"):
        for prompt in (32, 128):
            r = serving(f"p4-{step}-decode-{dtype}.json", prompt, 256)
            old = prior[dtype, prompt]
            rows.append(
                [
                    step,
                    dtype,
                    prompt,
                    f"{r['rate']:,.0f}",
                    f"{(r['rate'] / old - 1) * 100:+.2f}%",
                    f"[receipt]({r['receipt']})",
                ]
            )
            prior[dtype, prompt] = r["rate"]
table(
    [
        "Step",
        "State",
        "Prompt / batch256",
        "tok/s",
        "Delta vs preceding step",
        "Receipt",
    ],
    rows,
)
emit(
    "The first step is compared with the previous P3 default; following "
    "steps are measured ablations, even though `combined` failed its self "
    "gate. Previous default receipts are `decode-before-{fp32,fp16}.json`; "
    "rounded measured baseline prompt32 rates were 17,512 / 19,805 tok/s, "
    "prompt128 17,106 / 19,178 tok/s."
)
emit()
rows = []
for step in ("combined", "joined", "gated", "tuned", "stable"):
    rows.append(
        [step]
        + [
            f"{serving(f'p4-{step}-prefill-{dtype}.json', 2048, 16)['rate']:,.0f}"
            for dtype in ("fp32", "fp16")
        ]
    )
table(["Step / prefill16×2048", "FP32 input tok/s", "FP16 input tok/s"], rows)
emit(
    "Previous matched prefill was 59,070 / 59,916 input tok/s (FP32/FP16), "
    "from `prefill-matched-{fp32,fp16}.json`. Prefill step receipts are "
    "`p4-{step}-prefill-{fp32,fp16}.json`."
)
emit()
rows = []
for step in ("combined", "joined", "gated", "tuned", "stable"):
    d = read(f"p4-{step}-screen.json")
    sh = read(f"p4-{step}-shadow.json")
    rows.append(
        [
            step,
            f"{d['summary']['top1_agreement'] * 100:.4f}%",
            f"{d['summary']['mean_abs_dlogprob']:.6f}",
            f"{d['short_nll_delta']:+.6f}",
            "/".join(f"{r['delta']:+.6f}" for r in d["long_nll"]),
            f"{d['self_max']:.6f}",
            d["requested_teacher_nll_self_screen_pass"],
            sh["tokens_identical"] and sh["logprobs_exact"],
        ]
    )
table(
    [
        "Step",
        "Teacher top1",
        "Mean |dlp|",
        "Short NLL delta",
        "Each long NLL delta",
        "Self max",
        "Teacher/NLL/self pass",
        "Shadow exact",
    ],
    rows,
)
emit(
    "Screens retain the P2b limits: teacher top1 ≥94.2264%, mean |dlp| "
    "≤0.084353, absolute short/each-long NLL delta ≤0.01 nats, self max "
    "≤0.578507. Source receipts are `p4-{step}-screen.json`, "
    "`p4-{step}-parity.json`, and `p4-{step}-{plain,shadow}.json`. No "
    "threshold was relaxed to pass a candidate."
)
emit()
emit(
    "Implementation: (1) combine existing pointwise fusion with "
    "BV64/four-warps decode; (2) use existing `join_linears` for "
    "q/k/v/e0/f0/b/c/g0, preserve b/c biases and split outputs as views; "
    "batch e1/f1 with a two-bank GEMM, leave differently shaped g1 "
    "separate; (3) fold L2 q/k/e normalization, per-channel decay, "
    "beta/gamma sigmoid into recurrence, retaining BF16 normalization "
    "rounding and FP32 arithmetic; (4) tune tiles/warps/grid with CUPTI "
    "graph/cold-L2 timing. Convolution remains separate, as in Kappa. "
    "Output RMSNorm/gate remains one separate fused kernel."
)
emit()
emit(
    "The combined self failure was reproduced at sequence 2 / generated "
    "position 103: 0.6153526 nats. The same saved history scored with BF16 "
    "experts reduces its maximum to 0.3044996 nats "
    "(`p4-combined-self-{fp4,bf16}.json`), showing a contribution from "
    "full-prefill expert-path differences. It remains rejected. Batch-32 "
    "continuation self consistency differs with batch composition; this "
    "does not override the failed eight-sequence step screen."
)
emit()
emit(
    "The 96-config sweep covers batches 1/4/256, both states, "
    "BV16/32/64/128, four/eight warps and sequence/head grid order. Inline "
    "reduction changes a few normalized q/k values by one BF16 step in the "
    "large seeded case; captured gate outputs localize this difference. "
    "Gates and end-to-end output retain their explicit tolerances; "
    "recurrence is also compared against those captured gate values at the "
    "original state tolerance. Per-config numerical diagnostics and "
    "timings are in `p4-gated-sweep.json`; failed diagnostic logs are "
    "preserved. This is tolerance-based kernel correctness, not bit "
    "equality with separate preparation."
)
emit()
rows = []
for dtype in ("torch.float32", "torch.float16"):
    for batch in (1, 4, 256):
        best = min(
            (
                r
                for r in read("p4-gated-sweep.json")["rows"]
                if r["dtype"] == dtype and r["batch"] == batch
            ),
            key=lambda r: r["us"],
        )
        rows.append(
            [
                dtype,
                batch,
                best["block_v"],
                best["num_warps"],
                best["grid_order"],
                f"{best['us']:.3f}",
            ]
        )
table(["State", "Batch", "BV", "Warps", "Grid order", "Kernel µs"], rows)
emit(
    "Tuned profile uses the measured small-batch BV32/head/four-warps "
    "winner, large-batch BV128/sequence/eight warps for FP32 and four "
    "warps for FP16. Generated PTX/TTGIR and resource metadata are "
    "preserved in `generated/` and `p4-generated-code.json`; selected "
    "kernels have no spills or local-memory PTX instructions."
)
emit()
emit("## 32-sequence greedy gate")
emit()
emit(
    "The original four combinations each failed a gate: combined fails "
    "self consistency; joined/gated/tuned fail the 32-sequence greedy "
    "envelope. The previous default also gives 11/32 under the new "
    "instrument, exceeding the 10/32 envelope. These outcomes remain "
    "failures in the receipts."
)
emit()
emit(
    "A real-activation diagnostic separates joined-GEMM rounding from "
    "second-stage batching (`p4-projection-rounding.json`). At batch32 "
    "joined addmm differs substantially from separate projections, while "
    "batched e/f expansions match. Explicit FP32 GEMM accumulation, with "
    "bias before BF16 storage, is much closer to the FP32 oracle "
    "(`p4-projection-stable.json`). The final `stable` candidate uses "
    "this one joined GEMM below 256 tokens, preserving the existing "
    "large-batch GEMM and expert policies. Its tile (32/64/128, four "
    "warps, three stages) comes from `p4-joined-gemm-sweep.json`, using "
    "CUPTI graph/cold-L2 timing. No threshold or expert precision "
    "switch is changed."
)
emit()
rows = []
for name, d in g["floors"].items():
    rows.append(
        [
            name,
            d["exact_matches"],
            d["large_margin_divergences"],
            d["large_margin_divergences_per_32"],
            f"{d['fraction'] * 100:.2f}%",
        ]
    )
for name, d in g["candidates"].items():
    rows.append(
        [
            name,
            d["exact_matches"],
            d["large_margin_divergences"],
            d["large_margin_divergences_per_32"],
            f"{d['fraction'] * 100:.2f}%",
        ]
    )
table(
    [
        "Floor/candidate",
        "Exact /32",
        "First divergence margin ≥0.1 count",
        "Normalized per32",
        "Fraction",
    ],
    rows,
)
emit(
    f"Acceptance envelope: max(F2,F3,F5) = **{g['floor_max_count']}/32**. "
    f"`{a.selected}` passes. Every row uses all 32 real prompts and 128 "
    f"generated tokens; fork candidate and F5 state storage is FP32. "
    f"This is first-divergence counting, not the number "
    f"of mismatching tokens. Full positions/margins are in "
    f"[p4-greedy32-verdict.json](p4-greedy32-verdict.json). R_bf16 and F2 "
    f"reuse their original eight traces and generate the missing 24 "
    f"individually; shards change GPU assignment only. F3 is a genuine mesh "
    f"batch32 forward and F5 is fork max_num_seqs 1 versus 32. "
    f"Hash-preserving assembled source receipts are "
    "`p4-reference-{bf16,fla}-32.json`, and the batch receipt is "
    "`p4-reference-batch32-32.json`. Fork continuations are "
    "`p4-floor-{single,batch32}.json` and `p4-{step}-greedy32.json`."
)
emit()
emit(
    "The final batch32 continuation self check also passes: maximum "
    "0.5372305 nats, compared with the preserved 0.578507 limit. The "
    "step's original eight-sequence self maximum is 0.5403609 nats. "
    "Receipt: `p4-stable-greedy32.json`. FP16 state is separately "
    "qualified by forced cached-decode NLL and the final exact shadow; "
    "the 32-sequence greedy count is not a FP16-specific claim."
)
emit()
emit("## Final default decode")
emit()
rows = []
for prompt in (32, 128):
    for batch in (1, 4, 256):
        pair = [
            serving(f"p4-final-decode-{dtype}.json", prompt, batch)
            for dtype in ("fp32", "fp16")
        ]
        rows.append(
            [prompt, batch]
            + [f"{r['rate']:,.0f}" for r in pair]
            + [f"{r['mean_itl_ms']:.3f}" for r in pair]
        )
table(
    [
        "Prompt",
        "Batch",
        "FP32 tok/s",
        "FP16 tok/s",
        "FP32 mean ITL ms",
        "FP16 mean ITL ms",
    ],
    rows,
)
emit(
    "Receipts: [FP32](p4-final-decode-fp32.json), "
    "[FP16](p4-final-decode-fp16.json). ITL is the median of trial "
    "per-request mean inter-token latency. Actual state overrides are "
    "retained separately from the source model profile config. The "
    "in-place sampler flag is enabled, but Lambda's BF16 logits require "
    "a FP32 copy: its measured reused_logits_calls counter is zero. "
    "Kappa's preserved FP8-head lane records actual reuse. Slot-cache "
    "counters confirm reuse in Lambda. No sampler-reuse speedup is claimed."
)
emit()
emit("## Final matched prefill")
emit()
rows = []
for prompt in (128, 2048):
    for batch in (1, 16):
        rows.append(
            [prompt, batch]
            + [
                format(
                    serving(f"p4-final-prefill-{dtype}.json", prompt, batch)["rate"],
                    ",.0f",
                )
                for dtype in ("fp32", "fp16")
            ]
        )
table(["Prompt", "Batch", "FP32 input tok/s", "FP16 input tok/s"], rows)
emit(
    "Receipts: [FP32](p4-final-prefill-fp32.json), [FP16](p4-final-prefill-fp16.json)."
)
emit()
emit("## Previous default and unchanged same-box Kappa")
emit()
rows = []
for workload, prompts, batches in (
    ("decode", (32, 128), (1, 4, 256)),
    ("prefill", (128, 2048), (1, 16)),
):
    for prompt in prompts:
        for batch in batches:
            for dtype in ("fp32", "fp16"):
                previous = preserved_rate(
                    f"decode-before-{dtype}.json"
                    if workload == "decode"
                    else f"prefill-matched-{dtype}.json",
                    prompt,
                    batch,
                )
                kappa = preserved_rate(
                    f"kappa-sameBox-{workload}-{dtype}.json", prompt, batch
                )
                final = serving(f"p4-final-{workload}-{dtype}.json", prompt, batch)[
                    "rate"
                ]
                rows.append(
                    [
                        workload,
                        prompt,
                        batch,
                        dtype,
                        f"{previous:,.0f}",
                        f"{final:,.0f}",
                        f"{kappa:,.0f}",
                        f"{100 * (final / previous - 1):+.2f}%",
                        f"{100 * (final / kappa - 1):+.2f}%",
                    ]
                )
table(
    [
        "Workload",
        "Prompt",
        "Batch",
        "State",
        "Previous Lambda tok/s",
        "Final Lambda tok/s",
        "Kappa tok/s",
        "Final vs previous",
        "Final vs Kappa",
    ],
    rows,
)
emit(
    "Decode rates are cached output tok/s; prefill rates are input tok/s. "
    "Kappa receipts are the preserved `kappa-sameBox-{decode,prefill}-"
    "{fp32,fp16}.json` from P3. The remaining batch256 decode difference "
    "depends on state dtype and prompt length; final FP16 prompt128 "
    "can exceed the measured Kappa rate. Small-batch decode remains "
    "materially slower than Kappa: 27.23–29.97% at batch1 and "
    "17.64–20.48% at batch4 across these measured cases. These are observed "
    "rates across separate runs, not a claim of statistical equivalence."
)
emit()
emit("## Simultaneous DP8")
emit()
previous_dp8 = read("p3-summary.json")["dp8"]
rows = []
for r in s["dp8"]:
    previous = next(
        old["cohort_decode_rate_median"]
        for old in previous_dp8
        if old["state"] == r["state"] and old["prompt"] == r["prompt"]
    )
    final = r["cohort_decode_rate_median"]
    rows.append(
        [
            r["state"],
            r["prompt"],
            f"{previous:,.0f}",
            f"{final:,.0f}",
            f"{100 * (final / previous - 1):+.2f}%",
            f"{r['sum_per_gpu_medians']:,.0f}",
        ]
    )
table(
    [
        "State",
        "Prompt",
        "Previous cohort tok/s",
        "Final cohort tok/s",
        "Final vs previous",
        "Final sum per-GPU medians tok/s",
    ],
    rows,
)
emit(
    "All eight physical GPUs run independent TP=1 engines at batch256 "
    "simultaneously. Trials use a fresh filesystem barrier. Cohort rate is "
    "8×256×127 divided by latest last-token timestamp minus earliest "
    "first-token timestamp across the eight processes. Per-GPU receipts "
    "are `p4-final-dp8-{fp32,fp16}-gpu{0..7}.json`; every trial and both "
    "aggregate definitions are in [p4-summary.json](p4-summary.json). This "
    "is measured DP throughput; no 8× single-GPU estimate is substituted. "
    "Previous cohorts are preserved in `p3-summary.json` and "
    "`dp8-{fp32,fp16}-gpu{0..7}.json`. TP>1 remains unqualified."
)
emit()
emit("## Final forced cached-decode NLL")
emit()
rows = []
for name in ("windows", "long", "all"):
    r = nll[name]
    rows.append(
        [
            name,
            r["tokens"],
            f"{r['reference_nll']:.6f}",
            f"{r['decode_fp32_nll']:.6f}",
            f"{r['decode_fp16_nll']:.6f}",
            f"{r['fp16_degradation']:+.6f}",
        ]
    )
table(
    [
        "Corpus",
        "Cached positions",
        "R_bf16 NLL",
        "Decode FP32 NLL",
        "Decode FP16 NLL",
        "FP16 degradation",
    ],
    rows,
)
emit(
    f"FP16 qualifies on every case at ≤0.01 nats; maximum per-case "
    f"degradation is "
    f"{max(r['fp16_degradation'] for r in nll['cases']):+.6f} nats. "
    f"[Verdict](p4-decode-nll-final-verdict.json). Each case pre-fills 64 "
    f"real tokens, forces the ground-truth continuation, validates forced "
    f"IDs, and scores retained original logits with `raw_logprobs`. The "
    f"first prediction comes from prefill and is excluded. This qualifies "
    f"recurrent storage drift in the one-request BF16-expert decode path; "
    f"it is not a batch256 FP4 expert language-quality evaluation."
)
emit()
emit("## Final component profile versus Kappa")
emit()
rows = []
for name in ("p4-final-profile-fp32", "p4-final-profile-fp16", "profile-kappa"):
    d = read(name + ".json")
    for r in d["rows"]:
        times = r["component_timing_trials"][0][0]["last_step_gpu_ms"]

        def total(needle, times=times):
            return sum(
                v for k, v in times.items() if k.startswith("layer.") and needle in k
            )

        rows.append(
            [
                name,
                r["prompt_tokens"],
                r["concurrency"],
                f"{total('EDALayer') + total('GDN2Layer'):.3f}",
                f"{sum(v for k, v in times.items() if k.startswith('mixer.')):.3f}",
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
        "Projection ms (nested)",
        "MoE ms",
        "Attention ms",
        "Model ms",
    ],
    rows,
)
emit(
    "The EDA projection column includes the batched erase/decay module, in "
    "addition to the joined input, output gate expansion and output "
    "projection. It must not be added to the already inclusive mixer "
    "column. Kernel names/call counts are in "
    "`p4-final-profile-kernels-{fp32,fp16}.json`; same-box Kappa is the "
    "preserved FP32-state P3 `profile-kappa.json`, not a modified Kappa run. Baseline "
    "Lambda diagnostic mixer/projection/MoE totals were "
    "20.850/5.607/16.880 ms versus Kappa 9.155/1.441/14.387 ms "
    "(`profile-before-decode.json`)."
)
emit()
emit(
    "Final FP32 decode mixer time is 11.383 ms versus the previous "
    "20.850 ms and Kappa's 9.155 ms; nested projection time falls from "
    "5.607 to 2.807 ms (Kappa 1.441 ms). MoE remains 17.001 ms versus "
    "Kappa 14.387 ms, consistent with Lambda's additional shared expert. "
    "FP16 storage reduces Lambda mixer time to 9.543 ms. These are "
    "measured diagnostic component costs; state dtypes must be retained "
    "when comparing them."
)
emit()
emit(
    "The joined input removes seven projection launches per EDA layer; "
    "batching equal-rank e/f expansions removes another GEMM call, "
    "yielding four projection calls per EDA layer including o_proj. EDA "
    "retains an additional erase read/update, per-channel decay, three "
    "normalization reductions and its separate convolution/output gate. "
    "Lambda also has a shared expert absent from Kappa. The final "
    "component measurements quantify the remaining costs; eager event sums "
    "do not explain headline graph latency additively."
)
emit()
emit("## Validation, boundaries and open issues")
emit()
emit(
    "72 final focused tests passed across two runs: 49 EDA kernel tests "
    "and 23 loader/profile/scorer checks. Earlier runs remain separate. "
    "Tests cover both state dtypes, BV128, "
    "four/eight warps, head/sequence grid order, invalid/padded slots and "
    "strided/offset inputs and joined bias rounding. Logs: "
    "`p4-tests-stable.log`, `p4-tests-profile-final.log`. The 96-case sweep "
    "checks numerical outputs and state before timing. Each step has an "
    "exact scheduler shadow receipt, and final FP32/FP16 shadows both "
    "match tokens and logprobs exactly. Ruff check/format and shell syntax "
    "logs are preserved. Receipt-specific hashes and "
    "`P4_SOURCE_SHA256.json` retain final implementation identity."
)
emit()
emit(
    "Boundary incident: a newly launched fork quality job briefly "
    "overlapped the still-loading shadow job on GPU3; it was stopped and "
    "relaunched only after that GPU was idle. Its aborted log contributes "
    "no result. The first CUPTI callback had positional bindings "
    "inconsistent with the helper's input_args and was corrected before "
    "accepted timing; diagnostic logs preserve that failure. Transient "
    "allocator warnings during expert packing are distinguished from "
    "failed serving trials. Prefix mesh jobs were stopped by their exact "
    "recorded PIDs after saving sequences 0–15; independent reference "
    "shards supply 16–31. No sealed reference source/environment, main "
    "checkout, other host, commit or push was touched. All remote "
    "outputs/caches remain under `/root/vllm_eda`."
)
emit()
emit(
    "Open issues: broad language evaluation and actual all-batch FP4 "
    "teacher parity remain outside this qualification; recurrent FP16 "
    "quality is a corpus screen. TP>1 is unqualified. Exact mesh bit "
    "parity is not claimed: GEMM/batch/recurrent normalization arithmetic "
    "produces observable execution noise, whose greedy floor is now "
    "measured over 32 sequences. The combined-only candidate remains "
    "rejected for self consistency. Kappa's existing profile is unchanged."
)
a.output.write_text("\n".join(lines) + "\n")
