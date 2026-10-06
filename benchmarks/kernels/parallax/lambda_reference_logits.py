# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Save mesh bench-lane Lambda logits and repeated-forward greedy references.

Run with the mesh venv and mesh source on PYTHONPATH. No vLLM import is needed.
The receipt distinguishes a missing verification corpus from the original IDs.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).with_name("verification_inputs.json"),
    )
    parser.add_argument("--allow-deterministic-corpus", action="store_true")
    parser.add_argument("--backend", default="flashqla_eda")
    parser.add_argument("--expert-format", choices=("native", "bf16"), default="native")
    parser.add_argument("--decode-prompts", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--long-prompts", type=int, default=3)
    parser.add_argument("--long-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3-8B")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--self-floor", type=Path)
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--reuse-greedy", type=Path)
    parser.add_argument("--greedy-start", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg_doc = json.loads((args.export / "model_config.json").read_text())
    if args.inputs.exists():
        corpus = json.loads(args.inputs.read_text())
        prompts = corpus["prompts"]
        if not prompts or any(
            not isinstance(p, list) or len(p) < 2 or any(type(t) is not int for t in p)
            for p in prompts
        ):
            raise ValueError("Expected {'prompts': [[integer token IDs, ...], ...]}")
        corpus_info = {
            "path": str(args.inputs),
            "sha256": digest(args.inputs),
            "substituted": False,
        }
    elif args.allow_deterministic_corpus:
        prompts = [
            [128000] + [100 + (i * 7919 + j * 104729) % 127000 for j in range(127)]
            for i in range(4)
        ]
        corpus_info = {
            "path": str(args.inputs),
            "substituted": True,
            "reason": "verification_inputs.json absent",
            "recipe": "BOS 128000; 100+(i*7919+j*104729)%127000",
        }
    else:
        raise FileNotFoundError(args.inputs)
    prompts = [p[:128] for p in prompts]
    for p in prompts:
        if min(p) < 0 or max(p) >= cfg_doc["vocab_size"]:
            raise ValueError("Token outside export vocabulary")
    corpus_info["token_sha256"] = hashlib.sha256(
        json.dumps(prompts).encode()
    ).hexdigest()
    torch.manual_seed(0)
    torch.set_num_threads(16)
    from ops.eval.bench_saves import build_model, execution_record

    started = time.perf_counter()
    model, cfg, dev = build_model(
        args.export,
        device="cuda",
        model_config=args.export / "model_config.json",
        eda_kernel_backend=args.backend,
        step=21067,
        expert_format=args.expert_format,
    )
    torch.accelerator.synchronize()
    receipt = {
        "status": "running",
        "measurement_label": "MEASURED",
        "argv": sys.argv,
        "script_sha256": digest(__file__),
        "export": str(args.export),
        "manifest_sha256": digest(args.export / "manifest.json"),
        "model_config_sha256": digest(args.export / "model_config.json"),
        "corpus": corpus_info,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "device": str(dev),
        "build_seconds": time.perf_counter() - started,
        "execution": execution_record(model, args.expert_format),
        "swa_windows": {
            str(i): layer.sliding_window_size
            for i, layer in enumerate(model.layers)
            if getattr(layer, "sliding_window_size", None) is not None
        },
        "msa_dense_warmup": {
            str(i): bool(layer._dense_warmup_active)
            for i, layer in enumerate(model.layers)
            if hasattr(layer, "_dense_warmup_active")
        },
        "effective_logit_scale": float(model.output_scale().item()),
        "parameter_dtype_counts": {},
        "teacher": [],
        "greedy": [],
        "batch_size": args.batch_size,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("MESH_OFFLOAD_", "MESH_MSA_"))
            or k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH")
        },
    }
    for param in model.parameters():
        key = str(param.dtype)
        receipt["parameter_dtype_counts"][key] = (
            receipt["parameter_dtype_counts"].get(key, 0) + param.numel()
        )

    def write_receipt():
        path = args.output / "receipt.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(receipt, indent=2))
        tmp.replace(path)

    @torch.inference_mode()
    def forward(ids):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model(ids)
        if not torch.isfinite(result).all():
            raise RuntimeError("Nonfinite reference logits")
        return result

    if args.trace_only:
        trace = {}
        components = {}
        handles = []
        for name, module in model.layers[1].named_modules():
            if name in (
                "norm",
                "router",
                "latent_down",
                "latent_up",
                "shared_experts.0.up_proj",
                "shared_experts.0.down_proj",
            ):

                def component_hook(module, inputs, output, label=name):
                    components[label] = {
                        "input": inputs[0].detach().cpu(),
                        "output": [x.detach().cpu() for x in output]
                        if isinstance(output, tuple)
                        else output.detach().cpu(),
                    }

                handles.append(module.register_forward_hook(component_hook))
        for i, layer in enumerate(model.layers):

            def hook(module, inputs, output, index=i):
                trace[str(index)] = {
                    "input": inputs[0].detach().cpu(),
                    "output": output.detach().cpu(),
                }

            handles.append(layer.register_forward_hook(hook))
        ids = torch.tensor([prompts[0]], device=dev, dtype=torch.long)
        logits = forward(ids)
        torch.save(
            {
                "layers": trace,
                "logits": logits.cpu(),
                "moe_components": components,
                "moe_weights": {
                    k: v.detach().cpu() for k, v in model.layers[1].named_parameters()
                },
            },
            args.output / "trace.pt",
        )
        for handle in handles:
            handle.remove()
        receipt.update(status="complete", trace=str(args.output / "trace.pt"))
        write_receipt()
        return

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        receipt["tokenizer"] = args.tokenizer
    except (OSError, ValueError) as exc:
        receipt["tokenizer_unavailable"] = str(exc)
    write_receipt()
    if args.self_floor:
        receipt["self_consistency"] = []
        for i in range(args.decode_prompts):
            data = torch.load(args.self_floor / f"greedy_{i:03d}.pt", weights_only=True)
            ids = torch.cat((data["prompt_ids"], data["tokens"]))
            full = forward(ids[None].to(dev))[0].float().log_softmax(-1)
            positions = data["positions"].to(dev)
            tokens = data["tokens"].to(dev)
            chosen = full[positions, tokens].cpu()
            delta = (chosen - data["chosen_logprobs"]).abs()
            predictions = full[positions].argmax(-1).cpu()
            first = next(
                (
                    j
                    for j, (a, b) in enumerate(
                        zip(predictions.tolist(), data["tokens"].tolist(), strict=True)
                    )
                    if a != b
                ),
                None,
            )
            # Exact L versus L+128 comparison, on all shared positions.
            length = len(data["prompt_ids"])
            prefix = forward(ids[None, :length].to(dev))[0].float().log_softmax(-1)
            selected = prefix.argmax(-1)
            fixed_delta = (
                prefix.gather(1, selected[:, None]).squeeze(1)
                - full[:length].gather(1, selected[:, None]).squeeze(1)
            ).abs()
            receipt["self_consistency"].append(
                {
                    "sequence": i,
                    "prefix_generation_vs_full": {
                        "count": len(tokens),
                        "top1_agreement": float(
                            (predictions == data["tokens"]).float().mean()
                        ),
                        "max": float(delta.max()),
                        "mean": float(delta.mean()),
                    },
                    "conditional_greedy": {
                        "exact_match": first is None,
                        "first_divergence": first,
                        "reference_top2_margin": None
                        if first is None
                        else float(
                            data["top20_logprobs"][first, 0]
                            - data["top20_logprobs"][first, 1]
                        ),
                        "scope": "full teacher forcing on reference greedy history",
                    },
                    "L_vs_L_plus_128": {
                        "L": length,
                        "count": length,
                        "top1_agreement": float(
                            (selected == full[:length].argmax(-1)).float().mean()
                        ),
                        "max": float(fixed_delta.max()),
                        "mean": float(fixed_delta.mean()),
                    },
                }
            )
            write_receipt()
        receipt.update(status="complete")
        write_receipt()
        return
    cases = [(f"prompt_{i:03d}", p) for i, p in enumerate(prompts)]
    text_ids = [token for prompt in prompts for token in prompt]
    for i, length in enumerate((2048, 3072)[: args.long_prompts]):
        if len(text_ids) < length:
            raise ValueError("Long cases require enough real corpus tokens")
        cases.append((f"long_{i:03d}", text_ids[:length]))
    if args.skip_teacher:
        cases = []
    if args.batch_size > 1 and not args.skip_teacher:
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            ids = torch.tensor(batch, device=dev, dtype=torch.long)
            batch_logits = forward(ids)
            for offset, p in enumerate(batch):
                name = f"prompt_{start + offset:03d}"
                logits = batch_logits[offset].float()
                loss = F.cross_entropy(logits[:-1], ids[offset, 1:]).item()
                path = args.output / f"{name}.pt"
                torch.save(
                    {
                        "input_ids": ids[offset].cpu(),
                        "logits": logits.cpu(),
                        "top1": logits.argmax(-1).cpu(),
                        "next_token_loss": loss,
                    },
                    path,
                )
                receipt["teacher"].append(
                    {"name": name, "path": str(path), "length": len(p), "loss": loss}
                )
            del batch_logits, logits
            write_receipt()
        cases = [(name, p) for name, p in cases if name.startswith("long")]
    for name, p in cases:
        begin = time.perf_counter()
        ids = torch.tensor([p], device=dev, dtype=torch.long)
        logits = forward(ids)[0].float()
        loss = F.cross_entropy(logits[:-1], ids[0, 1:]).item()
        top1 = logits.argmax(-1).cpu()
        path = args.output / f"{name}.pt"
        torch.save(
            {
                "input_ids": ids[0].cpu(),
                "logits": logits.cpu(),
                "top1": top1,
                "next_token_loss": loss,
            },
            path,
        )
        del logits
        torch.accelerator.synchronize()
        receipt["teacher"].append(
            {
                "name": name,
                "path": str(path),
                "length": len(p),
                "loss": loss,
                "seconds": time.perf_counter() - begin,
            }
        )
        write_receipt()
        print(json.dumps(receipt["teacher"][-1]), flush=True)
    if args.batch_size > 1:
        ids = torch.tensor(prompts[: args.batch_size], device=dev, dtype=torch.long)
        records = [[] for _ in range(args.decode_prompts)]
        for j in range(args.decode_tokens):
            lp = forward(ids)[:, -1].float().log_softmax(-1)
            values, indices = lp.topk(args.top_k)
            token = lp.argmax(-1)
            for i in range(args.decode_prompts):
                records[i].append((token[i].cpu(), values[i].cpu(), indices[i].cpu()))
            ids = torch.cat((ids, token[:, None]), dim=1)
            print(f"batched greedy: {j + 1}/{args.decode_tokens}", flush=True)
        for i, record in enumerate(records):
            tokens = torch.stack([x[0] for x in record])
            path = args.output / f"greedy_{i:03d}.pt"
            torch.save(
                {
                    "prompt_ids": torch.tensor(prompts[i]),
                    "tokens": tokens,
                    "positions": torch.arange(127, 127 + args.decode_tokens),
                    "top20_ids": torch.stack([x[2] for x in record]),
                    "top20_logprobs": torch.stack([x[1] for x in record]),
                    "chosen_logprobs": torch.stack([x[1][0] for x in record]),
                },
                path,
            )
            receipt["greedy"].append(
                {
                    "name": f"greedy_{i:03d}",
                    "path": str(path),
                    "tokens": tokens.tolist(),
                }
            )
        write_receipt()
    for i, p in enumerate(
        prompts[: args.decode_prompts] if args.batch_size == 1 else []
    ):
        if i < args.greedy_start:
            continue
        if args.reuse_greedy and (args.reuse_greedy / f"greedy_{i:03d}.pt").exists():
            import shutil

            source = args.reuse_greedy / f"greedy_{i:03d}.pt"
            data = torch.load(source, weights_only=True)
            if (
                data["prompt_ids"].tolist() != p
                or len(data["tokens"]) != args.decode_tokens
            ):
                raise ValueError(f"Incompatible reused greedy artifact: {source}")
            path = args.output / source.name
            shutil.copyfile(source, path)
            receipt["greedy"].append(
                {
                    "name": source.stem,
                    "path": str(path),
                    "tokens": data["tokens"].tolist(),
                    "reused_from": str(source),
                    "sha256": digest(source),
                }
            )
            write_receipt()
            continue
        begin = time.perf_counter()
        ids = torch.tensor([p], device=dev, dtype=torch.long)
        tokens, top_ids, top_lp, chosen_lp = [], [], [], []
        for j in range(args.decode_tokens):
            logits = forward(ids)[0, -1].float()
            lp = logits.log_softmax(-1)
            values, indices = lp.topk(args.top_k)
            token = logits.argmax().reshape(1, 1)
            tokens.append(int(token.item()))
            top_ids.append(indices.cpu())
            top_lp.append(values.cpu())
            chosen_lp.append(lp[token.item()].cpu())
            ids = torch.cat((ids, token), dim=1)
            if j % 16 == 0:
                print(f"greedy {i}: {j + 1}/{args.decode_tokens}", flush=True)
        path = args.output / f"greedy_{i:03d}.pt"
        torch.save(
            {
                "prompt_ids": torch.tensor(p),
                "tokens": torch.tensor(tokens),
                "positions": torch.arange(len(p) - 1, len(p) - 1 + len(tokens)),
                "top20_ids": torch.stack(top_ids),
                "top20_logprobs": torch.stack(top_lp),
                "chosen_logprobs": torch.stack(chosen_lp),
            },
            path,
        )
        torch.accelerator.synchronize()
        row = {
            "name": f"greedy_{i:03d}",
            "path": str(path),
            "tokens": tokens,
            "seconds": time.perf_counter() - begin,
        }
        if tokenizer is not None:
            row["generated_text"] = tokenizer.decode(tokens)
        receipt["greedy"].append(row)
        write_receipt()
    receipt.update(
        status="complete",
        total_seconds=time.perf_counter() - started,
        peak_memory_bytes=torch.accelerator.max_memory_allocated(),
    )
    write_receipt()
    print(
        json.dumps({"status": "complete", "total_seconds": receipt["total_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
