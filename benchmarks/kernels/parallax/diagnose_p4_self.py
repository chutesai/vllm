# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Localize a candidate self-consistency failure on its exact saved history."""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--receipt", type=Path, required=True)
p.add_argument("--inputs", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--bf16-experts", action="store_true")
a = p.parse_args()
d = json.loads(a.receipt.read_text())
row = max(d["self_consistency"], key=lambda r: r["delta"]["max"])
i = row["sequence"]
ids = json.loads(a.inputs.read_text())["prompts"][i]
saved = d["greedy"][str(i)]
llm = LLM(
    model=d["model"],
    dtype="bfloat16",
    skip_tokenizer_init=True,
    max_model_len=4096,
    max_num_seqs=1,
    max_num_batched_tokens=4096,
    enable_prefix_caching=False,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    hf_overrides={
        "gdn2_state_storage": d["state_storage"],
        **({"single_sparse_fp4_components": 0} if a.bf16_experts else {}),
    },
)
try:
    out = llm.generate(
        [{"prompt_token_ids": ids + saved["tokens"]}],
        SamplingParams(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            prompt_logprobs=0,
            detokenize=False,
        ),
        use_tqdm=False,
    )[0]
    values = []
    for j, (token, original) in enumerate(
        zip(saved["tokens"], saved["logprobs"], strict=True)
    ):
        teacher = out.prompt_logprobs[len(ids) + j][token].logprob
        values.append(
            {
                "position": j,
                "token": token,
                "cached_logprob": original,
                "prefill_logprob": teacher,
                "abs_delta": abs(teacher - original),
            }
        )
    a.output.write_text(
        json.dumps(
            {
                "measurement_label": "MEASURED",
                "source": str(a.receipt),
                "sequence": i,
                "bf16_experts": a.bf16_experts,
                "original_self": row,
                "max": max(values, key=lambda r: r["abs_delta"]),
                "positions": values,
            },
            indent=2,
        )
    )
finally:
    llm.llm_engine.engine_core.shutdown()
