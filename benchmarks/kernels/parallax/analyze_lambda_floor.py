# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare sealed mesh artifacts or captured fork distributions without a model."""

import argparse
import json
from pathlib import Path

import torch


def summary(rows):
    count = sum(r["tokens"] for r in rows)
    return {
        "tokens": count,
        "top1_agreement": sum(r["top1_agreement"] * r["tokens"] for r in rows) / count,
        "mean_abs_dlogprob": sum(r["mean_abs_dlogprob"] * r["tokens"] for r in rows)
        / count,
        "max_abs_dlogprob": max(r["max_abs_dlogprob"] for r in rows),
    }


def nll_summary(rows):
    return {
        kind: sum(r["mean"] * r["tokens"] for r in rows if r["case"].startswith(kind))
        / sum(r["tokens"] for r in rows if r["case"].startswith(kind))
        for kind in ("prompt", "long")
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--teacher-only", action="store_true")
    p.add_argument("--greedy-only", action="store_true")
    p.add_argument("--greedy-prompts", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(4)
    fork = args.reference.is_file()
    if fork:
        ref_doc = json.loads(args.reference.read_text())
        cand_doc = json.loads(args.candidate.read_text())
        assert ref_doc["status"] == cand_doc["status"] == "complete"
        names = sorted(ref_doc["teacher_artifacts"])
    else:
        ref_doc = json.loads((args.reference / "receipt.json").read_text())
        cand_doc = json.loads((args.candidate / "receipt.json").read_text())
        if not args.teacher_only:
            assert ref_doc["status"] == cand_doc["status"] == "complete"
        assert ref_doc["corpus"]["token_sha256"] == cand_doc["corpus"]["token_sha256"]
        names = sorted(x.stem for x in args.reference.glob("prompt_*.pt"))
        names += sorted(x.stem for x in args.reference.glob("long_*.pt"))
    receipt = {
        "measurement_label": "MEASURED",
        "status": "complete",
        "scope": "teacher_only" if args.teacher_only else "teacher_and_greedy",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "teacher": [],
        "greedy": [],
        "nll_reference": [],
        "nll_candidate": [],
    }
    for name in [] if args.greedy_only else names:
        if fork:
            ref = ref_doc["teacher_artifacts"][name]
            cand = cand_doc["teacher_artifacts"][name]
            assert ref["input_ids"] == cand["input_ids"]
            top = ref["top1"]
            delta = torch.tensor(
                [
                    abs(b[str(t)] - a[str(t)])
                    for t, a, b in zip(
                        top, ref["logprobs"], cand["logprobs"], strict=True
                    )
                ]
            )
            agreement = sum(
                a == b for a, b in zip(top, cand["top1"], strict=True)
            ) / len(top)
        else:
            ref = torch.load(args.reference / (name + ".pt"), weights_only=True)
            cand = torch.load(args.candidate / (name + ".pt"), weights_only=True)
            assert torch.equal(ref["input_ids"], cand["input_ids"])
            a, b = [x["logits"][:-1].float().log_softmax(-1) for x in (ref, cand)]
            top = a.argmax(-1)
            delta = (a.gather(1, top[:, None]) - b.gather(1, top[:, None])).abs()
            agreement = float((top == b.argmax(-1)).float().mean())
            for label, data in [("reference", ref), ("candidate", cand)]:
                receipt["nll_" + label].append(
                    {"case": name, "tokens": len(top), "mean": data["next_token_loss"]}
                )
        receipt["teacher"].append(
            {
                "case": name,
                "tokens": len(top),
                "top1_agreement": agreement,
                "mean_abs_dlogprob": float(delta.mean()),
                "max_abs_dlogprob": float(delta.max()),
            }
        )
    if fork:
        receipt["nll_reference"] = ref_doc["nll"]
        receipt["nll_candidate"] = cand_doc["nll"]
    for i in range(0 if args.teacher_only else args.greedy_prompts):
        if fork:
            ref, cand = [doc["greedy"][str(i)] for doc in (ref_doc, cand_doc)]
            expected, generated = ref["tokens"], cand["tokens"]
        else:
            ref, cand = [
                torch.load(path / f"greedy_{i:03d}.pt", weights_only=True)
                for path in (args.reference, args.candidate)
            ]
            expected, generated = ref["tokens"].tolist(), cand["tokens"].tolist()
        first = next(
            (
                j
                for j, (a, b) in enumerate(zip(expected, generated, strict=True))
                if a != b
            ),
            None,
        )
        shared = 128 if first is None else first
        margin = None
        if first is not None:
            # Fork receipts inherit reference margins only for fork-vs-mesh comparisons.
            # The fork's runner saves top2 itself for fork-vs-fork floors.
            if fork:
                margin = ref.get("top2_margins", [None] * 128)[first]
            else:
                margin = float(
                    ref["top20_logprobs"][first, 0] - ref["top20_logprobs"][first, 1]
                )
        lpkey = "logprobs" if fork else "chosen_logprobs"
        drift = [
            abs(float(a) - float(b))
            for a, b in zip(ref[lpkey][:shared], cand[lpkey][:shared], strict=True)
        ]
        receipt["greedy"].append(
            {
                "sequence": i,
                "exact_match": first is None,
                "first_divergence": first,
                "reference_top2_margin": margin,
                "shared_prefix_tokens": shared,
                "shared_prefix_max_abs_dlogprob": max(drift, default=0),
                "shared_prefix_mean_abs_dlogprob": sum(drift) / max(1, len(drift)),
            }
        )
    receipt["short_summary"] = (
        None
        if args.greedy_only
        else summary([r for r in receipt["teacher"] if r["case"].startswith("prompt")])
    )
    receipt["long_summary"] = (
        None
        if args.greedy_only
        else summary([r for r in receipt["teacher"] if r["case"].startswith("long")])
    )
    receipt["greedy_exact_matches"] = sum(r["exact_match"] for r in receipt["greedy"])
    receipt["large_margin_divergences"] = sum(
        r["reference_top2_margin"] is not None and r["reference_top2_margin"] >= 0.1
        for r in receipt["greedy"]
    )
    receipt["greedy_sequences"] = len(receipt["greedy"])
    receipt["large_margin_divergences_per_32"] = (
        receipt["large_margin_divergences"] * 32 / len(receipt["greedy"])
        if receipt["greedy"]
        else None
    )
    if args.teacher_only:
        receipt["greedy_exact_matches"] = None
        receipt["large_margin_divergences"] = None
    receipt["nll"] = (
        {}
        if args.greedy_only
        else {
            label: nll_summary(receipt["nll_" + label])
            for label in ("reference", "candidate")
        }
    )
    receipt["nll"]["delta"] = {
        kind: receipt["nll"]["candidate"][kind] - receipt["nll"]["reference"][kind]
        for kind in (() if args.greedy_only else ("prompt", "long"))
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "short_summary",
                    "long_summary",
                    "nll",
                    "greedy_exact_matches",
                    "large_margin_divergences",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
