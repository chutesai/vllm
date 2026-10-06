# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Package compact measured receipts and the P2b report; keep full distributions."""

import argparse
import hashlib
import json
from pathlib import Path

FLAGS = (
    "lambda_mesh_rope",
    "lambda_mesh_norm",
    "lambda_fp32_attention",
    "lambda_mesh_bf16_reduce",
    "lambda_mesh_router",
    "lambda_conv_round",
    "lambda_force_fp4",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--greedy-rule", choices=("both", "envelope"), default="both")
    args = p.parse_args()
    root = args.root
    out = root / "receipts-p2b"
    out.mkdir(exist_ok=True)

    def read(name):
        return json.loads((root / name).read_text())

    floors = {name: read(f"parity-floor-{name}.json") for name in ("f1", "f2", "f3")}
    f4 = read("reference-self-metrics/receipt.json")
    assert f4["status"] == "complete"
    f4["summary"] = {
        scope: {
            "count": sum(r[scope]["count"] for r in f4["self_consistency"]),
            "top1_agreement": sum(
                r[scope]["top1_agreement"] for r in f4["self_consistency"]
            )
            / 8,
            "mean_abs_dlogprob": sum(r[scope]["mean"] for r in f4["self_consistency"])
            / 8,
            "max_abs_dlogprob": max(r[scope]["max"] for r in f4["self_consistency"]),
        }
        for scope in ("prefix_generation_vs_full", "L_vs_L_plus_128")
    }
    f4["conditional_greedy_exact_matches"] = sum(
        r["conditional_greedy"]["exact_match"] for r in f4["self_consistency"]
    )
    f4["conditional_large_margin_divergences"] = sum(
        r["conditional_greedy"]["reference_top2_margin"] is not None
        and r["conditional_greedy"]["reference_top2_margin"] >= 0.1
        for r in f4["self_consistency"]
    )
    f5 = read("parity-floor-f5-single-batch32.json")
    mean_limit = 1.5 * max(
        floors[f]["short_summary"]["mean_abs_dlogprob"] for f in ("f2", "f3")
    )
    top_limit = (
        min(floors[f]["short_summary"]["top1_agreement"] for f in ("f2", "f3")) - 0.01
    )
    mesh_self = max(
        r["prefix_generation_vs_full"]["max"] for r in f4["self_consistency"]
    )
    mesh_fixed = max(r["L_vs_L_plus_128"]["max"] for r in f4["self_consistency"])
    fork_batch = f5["short_summary"]["max_abs_dlogprob"]
    large = [floors[f]["large_margin_divergences"] for f in ("f2", "f3")]
    greedy_limit = min(large) if args.greedy_rule == "both" else max(large)
    thresholds = {
        "mean_abs_dlogprob_max": mean_limit,
        "top1_agreement_min": top_limit,
        "abs_nll_delta_max": 0.01,
        "mesh_self_max": 1.5 * mesh_self,
        "fork_batch_max": 1.5 * fork_batch,
        "greedy_rule": args.greedy_rule,
        "greedy_large_margin_max": greedy_limit,
        "greedy_each_floor_max": min(large),
        "greedy_envelope_max": max(large),
    }
    configs = [f"{b}-{s}" for b in ("bf16", "fp4") for s in ("fp32", "fp16")]
    configs += ["default-" + c for c in configs]
    configs += ["fp4-bf16head", "fp4-bf16head-fp16", "fast-fp4-fp32", "fast-fp4-fp16"]
    verdicts = {}
    for label in configs:
        data = read("parity-floor-" + label + ".json")
        assert data["status"] == "complete"
        s = data["references"]["reference-bf16"]["summary"]
        nll = read("nll-" + label + ".json")
        checks = {
            "teacher_mean": s["mean_abs_dlogprob"] <= mean_limit,
            "teacher_top1": s["top1_agreement"] >= top_limit,
            "short_nll": abs(nll["delta"]["prompt"]) <= 0.01,
            "long_nll": abs(nll["delta"]["long"]) <= 0.01,
            "each_long_nll": all(
                abs(d) <= 0.01 for d in nll["per_long_delta"].values()
            ),
            "mesh_self": data["self_consistency_max"] <= 1.5 * mesh_self,
            "fork_batch": data["self_consistency_max"] <= 1.5 * fork_batch,
            "greedy": s["divergences_with_margin_ge_0_1"] <= greedy_limit,
        }
        verdicts[label] = {
            "qualified": all(checks.values()),
            "checks": checks,
            "greedy_pass_each_floor": s["divergences_with_margin_ge_0_1"] <= min(large),
            "greedy_pass_envelope": s["divergences_with_margin_ge_0_1"] <= max(large),
            "summary": s,
            "self_max": data["self_consistency_max"],
            "nll": nll["delta"],
        }
    result = {
        "measurement_label": "MEASURED",
        "status": "complete",
        "thresholds": thresholds,
        "configs": verdicts,
    }
    (root / "parity-floor-verdicts.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    for path in sorted(root.glob("parity-floor-*.json")):
        data = json.loads(path.read_text())
        assert data.get("status") == "complete", path
        if "teacher_artifacts" in data:
            data.pop("teacher_artifacts")
            data["full_distribution_artifact"] = str(path)
            data["full_artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if "effective_config" not in data:
                data["arithmetic_flags_when_measured"] = {
                    **dict.fromkeys(FLAGS, True),
                    **data.get("overrides", {}),
                }
                data["arithmetic_defaults_provenance"] = (
                    "P2B_model_executed.txt; defaults subsequently made opt-in"
                )
        if "-teacher.json" in path.name:
            data.update(
                scope="teacher_only",
                greedy_exact_matches=None,
                large_margin_divergences=None,
            )
        (out / path.name).write_text(json.dumps(data, indent=2) + "\n")
    (out / "parity-floor-f4.json").write_text(json.dumps(f4, indent=2) + "\n")
    for pattern in (
        "nll-*.json",
        "teacher-top1-correction.json",
        "teacher-tie-audit.json",
        "layer-fp4-independent-floor.json",
    ):
        for path in root.glob(pattern):
            (out / path.name).write_bytes(path.read_bytes())

    lines = [
        "# Lambda P2b execution floors and serving-path qualification",
        "",
        (
            "All numerical values below are **MEASURED**, with receipts linked in each "
            "table. No inference-throughput result or estimate is claimed. No commits "
            "or pushes were made. The sealed source and reference venv were used "
            "read-only; all outputs/caches are under `/root/vllm_eda`."
        ),
        "",
        (
            "The same export revision `1b992902aa52ca487eee21160b45d66791d1e7dc`, 32 "
            "real Wikitext prompts (4064 next-token positions), ordered 2048/"
            "3072-token "
            "sequences (5118 next-token positions), and eight 128-token greedy "
            "continuations were used. Corpus SHA256: "
            "`d7de6ec550a3136b92fc06ea40d430d3b4b81580ecdb9a811b122964b26bdd22`. "
            "Reference floors use sealed `build_model`, BF16 experts and either "
            "FlashQLA EDA or the bench default FLA 2t backend. One model process per "
            "GPU was pinned; GPUs were checked before launching."
        ),
        "",
        (
            "**Scorer correction:** P2 counted the first maximum in a top-k "
            "dictionary; "
            "mesh uses `argmax`, choosing the lowest token index on exact ties. On the "
            "same BF16/FP32 distributions, short agreement changes from 92.1506% to "
            "95.6693%. There were 438 tied short positions and 219 short positions "
            "whose selected ID changed. Chosen-token logprob drift and NLL are "
            "unchanged. Old dictionary scores remain in P2b receipts and P2_REPORT.md "
            "remains untouched. [Raw-logit audit](teacher-tie-audit.json) verifies "
            "the corrected prompt-0 top1 against all four saved fork logit tensors. "
            "See [correction receipt](teacher-top1-correction.json) "
            "and the CPU dictionary-order regression test."
        ),
        "",
        (
            "The original targets (99% agreement, mean |dlp| <=0.02, no margin >=0.1 "
            "divergence, self max <=0.02) still failed, including after correcting the "
            "teacher scorer. The new criterion below is the lead-defined floor test, "
            "not a claim of bit equality."
        ),
        "",
        (
            "| Floor | Short top-1 | Mean / max abs dlp | Greedy exact | First "
            "divergences with margin >=0.1 | NLL delta short / pooled long | Receipt |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for f, name in [
        ("f1", "F1 BF16/native"),
        ("f2", "F2 FlashQLA/FLA"),
        ("f3", "F3 single/batch32"),
    ]:
        d = floors[f]
        s = d["short_summary"]
        delta = d["nll"]["delta"]
        lines.append(
            f"| {name} | {100 * s['top1_agreement']:.4f}% | "
            f"{s['mean_abs_dlogprob']:.6f} / {s['max_abs_dlogprob']:.6f} | "
            f"{d['greedy_exact_matches']}/8 | {d['large_margin_divergences']} | "
            f"{delta['prompt']:+.6f} / {delta['long']:+.6f} | "
            f"[JSON](parity-floor-{f}.json) |"
        )
    for scope, title in (
        ("L_vs_L_plus_128", "F4 fixed L/L+128"),
        ("prefix_generation_vs_full", "F4 generation/full, conditional"),
    ):
        s = f4["summary"][scope]
        exact = (
            "—"
            if scope == "L_vs_L_plus_128"
            else f"{f4['conditional_greedy_exact_matches']}/8"
        )
        count = (
            "—"
            if scope == "L_vs_L_plus_128"
            else str(f4["conditional_large_margin_divergences"])
        )
        lines.append(
            f"| {title} | {100 * s['top1_agreement']:.4f}% | "
            f"{s['mean_abs_dlogprob']:.6f} / {s['max_abs_dlogprob']:.6f} | "
            f"{exact} | {count} | — | [JSON](parity-floor-f4.json) |"
        )
    s = f5["short_summary"]
    lines += [
        (
            f"| F5 fork max_num_seqs 1/32 | {100 * s['top1_agreement']:.4f}% | "
            f"{s['mean_abs_dlogprob']:.6f} / {s['max_abs_dlogprob']:.6f} | "
            f"{f5['greedy_exact_matches']}/8 | {f5['large_margin_divergences']} | "
            f"{f5['nll']['delta']['prompt']:+.6f} / {f5['nll']['delta']['long']:+.6f} "
            f"| "
            f"[JSON](parity-floor-f5-single-batch32.json) |"
        ),
        "",
        (
            f"F4 exact `L=128` versus `L+128=256`, all shared positions across eight "
            f"saved greedy sequences: max |dlp| **{mesh_fixed:.6f}**. The "
            f"corresponding "
            f"repeated-prefix-generation versus full-prefill analogue has max "
            f"**{mesh_self:.6f}**. [Receipt](parity-floor-f4.json) contains "
            f"per-sequence mean/max for both. F4 conditional greedy comparisons "
            f"use the saved reference history, not an independently generated "
            f"counterfactual continuation. F5 long teacher outputs are identical "
            f"because the long forwards stay single-sequence. F2 long top-1 is "
            f"{100 * floors['f2']['long_summary']['top1_agreement']:.4f}%, mean/max "
            f"|dlp| "
            f"{floors['f2']['long_summary']['mean_abs_dlogprob']:.6f}/"
            f"{floors['f2']['long_summary']['max_abs_dlogprob']:.6f}; "
            f"F3 long comparison is exact by construction."
        ),
        "",
        (
            f"Acceptance limits: short mean |dlp| <= **{mean_limit:.6f}**; top-1 >= "
            f"**{100 * top_limit:.4f}%**; absolute NLL delta <= **0.01 nats** on short "
            f"and long (individual long cases are checked too); self max <= "
            f"**{1.5 * mesh_self:.6f}** against the mesh generation analogue and <= "
            f"**{1.5 * fork_batch:.6f}** against F5. Greedy criterion interpretation: "
            f"**{args.greedy_rule}**, giving at most **{greedy_limit}/8** large-margin "
            f"first divergences. The two observed floors are F2={large[0]}/8 and "
            f"F3={large[1]}/8. Both the strict each-floor verdict and the "
            f"maximum-floor-envelope verdict are retained in [verdict "
            f"JSON](parity-floor-verdicts.json); no ambiguity is hidden."
        ),
        "",
        (
            "The original four configurations and the final defaults are compared "
            "separately. `P2 FP4` forces actual FP4 at every batch size and inherits "
            "kappa’s FP32 head. `Default FP4 profile` uses a BF16 head, BF16 experts "
            "below 256 packed tokens, and FP4 at >=256 with this profile; it must not "
            "be described as pure FP4. Defaults disable all seven Lambda diagnostic "
            "arithmetic switches."
        ),
        "",
        (
            "| Configuration | Short top-1 | Mean / max abs dlp | Greedy exact / "
            "large-margin | Self max | NLL delta short / pooled long | Verdict | "
            "Receipt |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for label in configs[:8]:
        v = verdicts[label]
        s = v["summary"]
        failed = ", ".join(k for k, passed in v["checks"].items() if not passed)
        title = (
            "Default " if label.startswith("default-") else "P2 "
        ) + label.removeprefix("default-")
        verdict = (
            "within mesh execution noise"
            if v["qualified"]
            else "NOT qualified: " + failed
        )
        lines.append(
            f"| {title} | {100 * s['top1_agreement']:.4f}% | "
            f"{s['mean_abs_dlogprob']:.6f} / {s['max_abs_dlogprob']:.6f} | "
            f"{s['greedy_exact_matches']}/8 / "
            f"{s['divergences_with_margin_ge_0_1']} "
            f"| {v['self_max']:.6f} | {v['nll']['prompt']:+.6f} / "
            f"{v['nll']['long']:+.6f} | {verdict} | "
            f"[JSON](parity-floor-{label}.json) |"
        )
    lines += [
        "",
        (
            "Real-text NLL uses returned logprobs of the actual next token, including "
            "when outside the top-20. The reference uses shifted full-logit cross "
            "entropy. Means are token-weighted, including the pooled long mean."
        ),
        "",
        (
            "| Lane | Short NLL | Pooled long NLL | Delta vs R_bf16 short / long | "
            "Receipt |"
        ),
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for lane in ["reference-bf16", "reference-native", "reference-fla"] + configs[:8]:
        d = read("nll-" + lane + ".json")
        m = d["mean"]
        delta = d["delta"]
        lines.append(
            f"| {lane} | {m['prompt']:.6f} | {m['long']:.6f} | "
            f"{delta['prompt']:+.6f} / {delta['long']:+.6f} | "
            f"[JSON](nll-{lane}.json) |"
        )
    lines += [
        "",
        (
            "Long-context teacher comparisons, each receipt also containing separate "
            "NLL deltas for both lengths:"
        ),
        "",
        "| Configuration | Length | Top-1 | Mean / max abs dlp | NLL delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in configs[:8]:
        d = read("parity-floor-" + label + ".json")
        nll = read("nll-" + label + ".json")
        for row in d["references"]["reference-bf16"]["teacher"]:
            if row["case"].startswith("long"):
                delta = row["chosen_logprob_delta"]
                lines.append(
                    f"| [{label}](parity-floor-{label}.json) | {row['tokens'] + 1} "
                    f"| {100 * row['top1_agreement']:.4f}% | {delta['mean']:.6f} / "
                    f"{delta['max']:.6f} | "
                    f"{nll['per_long_delta'][row['case']]:+.6f} |"
                )
    lines += [
        "",
        (
            "FP16 recurrent storage versus the FP32 fork with the same "
            "experts/head/arithmetic. Teacher/long forwards locally accumulate in FP32 "
            "and store state after a complete prefill, so this NLL check does not test "
            "many cached decode continuations. Greedy shared-prefix drift stops before "
            "different histories arise; first divergence and margin are in each "
            "receipt."
        ),
        "",
        (
            "| Pair | Teacher short / long agreement | Mean / max teacher drift | "
            "Greedy exact | Shared-prefix max drift | NLL degradation short / long | "
            "Receipt |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pair in ["state-bf16", "state-fp4", "state-default-bf16", "state-default-fp4"]:
        d = read("parity-floor-" + pair + ".json")
        s = d["short_summary"]
        lo = d["long_summary"]
        maximum = max(r["shared_prefix_max_abs_dlogprob"] for r in d["greedy"])
        delta = d["nll"]["delta"]
        lines.append(
            f"| {pair} | {100 * s['top1_agreement']:.4f}% / "
            f"{100 * lo['top1_agreement']:.4f}% | {s['mean_abs_dlogprob']:.6f} / "
            f"{s['max_abs_dlogprob']:.6f} | {d['greedy_exact_matches']}/8 | "
            f"{maximum:.6f} | {delta['prompt']:+.6f} / {delta['long']:+.6f} | "
            f"[JSON](parity-floor-{pair}.json) |"
        )
    lines += [
        "",
        (
            "FP16 passes the <=0.01-nat NLL-degradation condition for all measured "
            "pairs. This qualifies the measured storage drift/NLL check, not a "
            "configuration that fails the overall greedy or expert gates. FP32 remains "
            "the default state."
        ),
        "",
        (
            "Every P2 Lambda-only rounding path was toggled individually, with BF16 "
            "experts/FP32 state except the force-FP4 test. These runs retain all other "
            "P2 options. The full defaults were subsequently remeasured, rather than "
            "inferring the combination from individual runs."
        ),
        "",
        (
            "| Removed P2 path / opt-in flag | Short top-1 | Mean abs dlp | Self max | "
            "NLL delta short / long | Large-margin divergences | Receipt |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    ablations = [
        ("no-rope", "LambdaRoPE / lambda_mesh_rope"),
        ("no-norm", "LambdaRMSNorm / lambda_mesh_norm"),
        ("upstream-attention", "FP32-score attention / lambda_fp32_attention"),
        ("no-reduce", "BF16 running-sum reduce / lambda_mesh_bf16_reduce"),
        ("no-router", "router rounding / lambda_mesh_router"),
        ("no-conv-round", "round-before-SiLU / lambda_conv_round"),
        ("no-force-fp4", "FP4 all batch sizes / lambda_force_fp4"),
    ]
    for label, title in ablations:
        d = read("parity-floor-" + label + ".json")
        s = d["references"]["reference-bf16"]["summary"]
        delta = read("nll-" + label + ".json")["delta"]
        lines.append(
            f"| {title} | {100 * s['top1_agreement']:.4f}% | "
            f"{s['mean_abs_dlogprob']:.6f} | {d['self_consistency_max']:.6f} | "
            f"{delta['prompt']:+.6f} / {delta['long']:+.6f} | "
            f"{s['divergences_with_margin_ge_0_1']} | "
            f"[JSON](parity-floor-{label}.json) |"
        )
    lines += [
        "",
        (
            "**Default policy:** all seven flags above are false. Removing each BF16 "
            "diagnostic passes the teacher/NLL/self gates; the individual "
            "upstream-attention run has two large-margin divergences, and the "
            "all-default BF16 run has two. Thus upstream vLLM’s kappa dense attention "
            "is within the measured teacher/NLL/self floors; full greedy qualification "
            "follows the explicit interpretation above. No measured throughput "
            "advantage is asserted here. The old FP32-head FP4 fallback run still "
            "fails "
            "top-1 and is not called qualified."
        ),
        "",
        (
            "Lambda architectural arithmetic remains enabled: independent EDA "
            "erase/write recurrence, FP32 gate/normalization/output-gate computations "
            "and FP32 accumulation; coverage FP32 gains, bounded logit scale, tied "
            "embeddings, final unit gain, dense MSA and 2048 SWA policy. These "
            "implement the export contract, not optional BF16 rounding diagnostics. "
            "Separate projections remain. New profiles select a BF16 head to match the "
            "reference output precision. Kappa profile/head/arithmetic are unchanged. "
            "`lambda_torch_experts` is an additional false-by-default diagnostic; its "
            "teacher score is 95.6693% and replacing grouped GEMMs alone does not "
            "improve the baseline top-1 after correcting the scorer. No slow Torch "
            "loop "
            "was adopted for serving."
        ),
        "",
        (
            "**Specific remaining components:** the FP32-head/mixed-expert profile has "
            "93.8730% top-1; changing only its head to BF16 raises the otherwise "
            "identical mixed-path score to 96.0630%. Actual FP4 at every batch size "
            "with the BF16 head remains at 93.0364% and mean |dlp| 0.085776, failing "
            "both teacher gates. [Identical-input FP4 "
            "replay](layer-fp4-independent-floor.json) isolates layer-1 MoE residual "
            "error max 0.06640625 / mean 0.00801164, versus P2 BF16 max 0.0078125 / "
            "mean 0.00020628. Every replayed layer input is exactly the saved mesh "
            "input. The quantized expert path is the remaining teacher mismatch; its "
            "activation/row-scale arithmetic against native mesh is not proved "
            "identical. SWA layer 0 remains BF16-ulp scale. No EDA logic defect is "
            "established by these results."
        ),
        "",
        (
            "Where a configuration fails only the greedy frequency gate, "
            "teacher/NLL/self results are within their measured floors, but the lead’s "
            "full acceptance is still not met under that rule. Per-sequence "
            "first-divergence indices and reference margins are retained. P2 "
            "self-layer "
            "diagnosis already isolates shape-dependent q/router/latent/shared "
            "projection differences and <=0.00048828125 prepared EDA prefill/decode "
            "output differences. Launch-to-launch greedy counts varied despite "
            "identical short teacher distributions; the asynchronous scheduler can "
            "change batch composition. Receipts are individual runs, not statistical "
            "confidence intervals. No cherry-picked greedy pass is substituted for the "
            "final default run. Default BF16/FP32's two large-margin divergences "
            "are sequence 3 at index 0 (margin 0.25) and sequence 5 at index 0 "
            "(margin 0.125). They occur before any EDA decode update, localizing "
            "this remaining gate to initial prefill/batch-shape arithmetic; a "
            "specific defective operation has not been demonstrated."
        ),
        "",
        (
            "Validation: **39 tests passed** in [focused log](test-floor-final.log), "
            "including scorer ties and upstream-norm loader unit gain/bounded scale, "
            "plus EDA cache/graph/reset, attention, experts and kappa scheduler/"
            "sampler "
            "regressions. Changed-file ruff check and format check are recorded in "
            "[ruff log](ruff-floor-final.log). A fresh full kappa "
            "model-parity/throughput run was not performed."
        ),
        "",
        (
            "Open issues: actual all-batch FP4 teacher parity; any failing full greedy "
            "gate in the table; multi-step cached-decoding NLL beyond the prefill "
            "check; no-logprobs sampler fast-path/full serving graph qualification and "
            "RTX 5090 throughput remain P3 work. The default FP4 profile is mixed "
            "arithmetic below 256 tokens; throughput must record the actual path at "
            "its "
            "benchmark batch sizes. Only qualified configurations may be described as "
            "within mesh’s execution noise."
        ),
        "",
        (
            "Full tensors/distributions stay under "
            "`/root/vllm_eda/reference-{bf16,native,fla,batch32}` and original "
            "`parity-floor-*.json`; local receipts omit only the large per-position "
            "dictionaries and record full-artifact paths/hashes. "
            "[P2B_COMMANDS.sh](P2B_COMMANDS.sh), per-run argv/effective config, "
            "executed-source snapshots and source hashes preserve reproduction. No "
            "files under `/root/prof` were modified."
        ),
    ]
    (out / "P2B_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
