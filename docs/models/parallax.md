# Kappa trained-model inference — 2026-10-03

This package contains the validated RTX 5090 runtime from the Kappa development
box, ported to a fresh upstream checkout. Historical receipts are kept separate.

## Installation

Qualified runtime: Linux, one RTX 5090 (SM120a), Python 3.12, CUDA 13.0 nvcc,
upstream vLLM `5f30fc7031cae49bf51073fc953d419b08f8887c`.
The fresh source build and its matching CUDA 13 precompiled artifacts are tested
separately from the historical vLLM 0.29 wheel receipts. The tested environment
uses Torch 2.13.0+cu132, Triton 3.7.1 and FlashInfer 0.7.0.post1; its full
package versions are in the fresh-upstream report directory. The Parallax CUDA extension compiles during installation; a CUDA compiler and
C++ compiler are required for the optimized expert path. Other architectures,
multi-GPU operation, speculative decoding and prefix caching are not qualified.
The integrated model has no dependency on Mesh or a separate plugin package.

From a checkout of this fork, install the fork into an isolated environment:

For a source copy without `.git` (for example, an rsync copy), first set
the version in the environment. Otherwise setuptools-scm cannot determine the
package version:

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION=0.30.1rc1.dev619+g5f30fc703.parallax
```

```bash
uv venv --python 3.12 .venv-kappa
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=5f30fc7031cae49bf51073fc953d419b08f8887c \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
  uv pip install --python .venv-kappa/bin/python -e . --torch-backend=auto
```

Parallax is registered in vLLM's built-in model and config registries. The fork
builds `_parallax_C` through CMake during installation, including when upstream
precompiled binaries are used. Parallax expert code does not invoke nvcc in workers.
Upstream components such as FlashInfer can still compile their own kernels. CUDA 13 and a C++
compiler are needed at build time for the sparse-FP4 path; BF16 expert execution
does not need the custom extension.

## Model preparation

Download a complete canonical compact export from
`chutesai/parallax-8b-kappa`. Select the export named by that snapshot's
`LATEST.json`; resolve and record the Hub revision before downloading. Do not mix
files from different exports. The receipt checkpoint was pinned at revision
`15975e88320522284396bfe15c02e533d8b49dc4`, step 37898 (541B export), rather than
silently changing checkpoints between benchmarks.

Convert the downloaded export directory containing `manifest.json`,
`coverage.json`, `layouts.json`, `model_config.json`, `indexer.bundle`,
`relay_pack/` and `packed_experts.tar`:

```bash
.venv-kappa/bin/python tools/convert_parallax_checkpoint.py \
  --export /absolute/path/to/canonical-export \
  --output /absolute/path/to/kappa-bf16
.venv-kappa/bin/python tools/prepare_parallax_profile.py \
  --checkpoint /absolute/path/to/kappa-bf16 \
  --output /absolute/path/to/kappa-sm120
```

The converter validates payload and tensor hashes, rejects nonzero expert carry,
and materializes sharded canonical BF16 safetensors. This intermediate is larger
than the compact export. The runtime then packs experts into its native layout.
The profile creates a new config plus symlinks to weights; it does not modify
source weights or their inference policy. It refuses unqualified geometry.
For tokenizer-based serving, install the tokenizer files specified by the export's
tokenizer provenance into the converted checkpoint directory. The benchmark uses
integer token IDs and does not require a tokenizer.

## Reproduce the best validated decode path

Run from a directory outside the fork (substitute absolute paths):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 \
  /absolute/path/to/fork/.venv-kappa/bin/python \
  /absolute/path/to/fork/benchmarks/kernels/parallax/bench_serving_fp4.py \
  --model /absolute/path/to/kappa-sm120 --output /absolute/path/to/decode.json \
  --engine-context 8192 --batches 256 --prompts 32,128 --tokens 128 \
  --repeats 5 --warmups 4 --max-batched-tokens 32768 \
  --gpu-memory-utilization 0.7 --temperature 0.8 --top-p 0.95 \
  --synchronize-arrivals --full-decode-graph --capture-sizes 1,4,256 \
  --decode-slot-cache --inplace-sampler
```

Historical vLLM 0.29 measurements: prompt-32 **23,809 decode tokens/s**;
prompt-128 **22,649 decode tokens/s**, concurrency 256, 8192 context reservation.
The native integration measured **23,853** and **22,810 decode tokens/s**,
respectively (median of five decode-only trial rates), with all five sampled-token
hashes per shape identical to the previous plugin port. Receipts:
`reports/parallax_5090/kappa_native_20261003/`. The intermediate plugin-port
receipts remain in `reports/parallax_5090/kappa_upstream_5f30_20261003/`.
These are aggregate decode-only rates, excluding prefill; they are not batch-one
latencies or total generation throughput. JSON receipts and matched baselines are
in `reports/parallax_5090/kappa_20261003/`. Workloads and engine reservations affect
absolute rates. The profile includes the selected sparse-FP4 expert kernel, FP32 GDN2 grid order 3,
chunked causal MSA scoring and bit-exact selection finalization.

A fresh prefill smoke benchmark measured **67,946 input tokens/s** at 16
concurrent requests with 2,048-token prompts and one output token (three trials,
one warmup), including first output and scheduler time. Its token-ID workload
is synthetic and does not establish long-context quality.

Dense weights/KV remain BF16, router parameters/logits and recurrent state remain
FP32, and the head uses BF16 operands with FP32 accumulation/output. Expert weights
remain ternary with FP32 row scales, stored in a 1.5-bit-per-logical-weight execution
layout. **Single-component FP4 expert activations are approximate relative to BF16**;
bit-exact optimization comparisons do not establish whole-model equivalence to a
BF16 reference. FP16 recurrent state and the numerically divergent projected router
are disabled. Rejected experimental FP4 kernels are not included in this fork.

The benchmark installs guarded scheduler and sampler optimizations through worker
RPCs. A plain `vllm serve` launch does not install those two RPC-driven optimizations.
Use the benchmark as the complete reproduction entry point; ordinary LLM/serving
can still use the model profile and standard upstream scheduler/sampler. Prefix
cache sharing and speculative decoding are unsupported by this runtime.

## Source and validation

This release integrates the validated implementation into the fork. Source remains
uncommitted for human review. Upstream base:
`5f30fc7031cae49bf51073fc953d419b08f8887c`.
The native wheel passes model/config/extension import checks outside the checkout
with Mesh and the previous plugin blocked. Small-batch pair-LUT inference matches
the prior port at batch sizes 1 and 4. The native receipt directory includes the
wheel SHA256 and a source manifest.
Historical receipts in `kappa_20261003` describe the previous runtime; fresh source
validation is recorded separately. No broad language-quality evaluation is implied
by matching tokens/logprobs to the same approximate expert profile.

Run the focused tests, then compare the standard and optimized scheduler on the
same runtime:

```bash
uv pip install --python .venv-kappa/bin/python pytest
.venv-kappa/bin/python -m pytest --confcutdir=tests/model_executor/model_loader \
  tests/model_executor/model_loader/test_parallax.py -q
.venv-kappa/bin/python -m pytest --confcutdir=tests/v1/sample \
  tests/v1/sample/test_parallax_sampler.py -q
.venv-kappa/bin/python -m pytest --confcutdir=tests/v1/core \
  tests/v1/core/test_parallax_decode.py -q
.venv-kappa/bin/python benchmarks/kernels/parallax/verify_inference.py \
  --model /absolute/path/to/kappa-sm120 \
  --inputs benchmarks/kernels/parallax/verification_inputs.json \
  --save-baseline /tmp/kappa-reference.pt --output /tmp/kappa-reference.json
.venv-kappa/bin/python benchmarks/kernels/parallax/verify_inference.py \
  --model /absolute/path/to/kappa-sm120 \
  --inputs benchmarks/kernels/parallax/verification_inputs.json \
  --baseline /tmp/kappa-reference.pt --output /tmp/kappa-optimized.json \
  --optimized --shadow
```

This numerical screen uses greedy decoding to make logprob comparisons meaningful.
The throughput benchmark above uses stochastic sampling. Shadow mode executes the
upstream allocation and checks each predicted cache no-op against it.

## Source layout

- Model: `vllm/model_executor/models/parallax.py`.
- Config and supported profile: `vllm/transformers_utils/configs/parallax*`.
- Canonical checkpoint mapping/codecs: `vllm/model_executor/model_loader/parallax*`.
- Experts: `vllm/model_executor/layers/fused_moe/parallax/`.
- GDN2: `vllm/model_executor/layers/mamba/gdn/parallax*`.
- Attention and BAR: `vllm/model_executor/layers/attention/`.
- Cache and sampler optimizations: `vllm/v1/core/sched/parallax_decode.py` and
  `vllm/v1/worker/gpu/parallax_sampler.py`.
- Compiled CUDA: `csrc/parallax/`, built by `cmake/parallax.cmake`.
- Conversion/profile utilities: `tools/`.
- Reproduction scripts: `benchmarks/kernels/parallax/`.

The old version-numbered FP4 candidates are removed. Legacy `v28` checkpoint
configs map to the descriptive `sparse_ternary_fp4` implementation. Retired
experimental settings fail explicitly instead of selecting unshipped code.

## Lambda trained-model inference — 2026-10-05

Lambda uses EDA recurrent mixers, dense causal MSA during this export's indexer
warmup, 2048-token SWA windows, one shared expert, tied embeddings and the export's
bounded learned logit scale. The Kappa measurements above remain historical Kappa
claims. The Lambda results below were measured on RTX 5090, 2026-10-05;
same-box Kappa was measured separately from the historical Kappa results.

Install from this fork's checkout into an isolated Lambda environment (a source
copy without `.git` also needs the version environment variable described above):

```bash
uv venv --python 3.12 .venv-lambda
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_COMMIT=5f30fc7031cae49bf51073fc953d419b08f8887c \
VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
  uv pip install --python .venv-lambda/bin/python -e . --torch-backend=auto
```

The measured box uses the fork editable install, Torch 2.13.0+cu132 and Triton 3.7.1.
Build the EDA dependency from the pinned FlashQLA source with TileLang 0.1.13.
**PENDING PUBLICATION:** the EDA branch is not yet public. The target repository is
`https://github.com/chutesai/FlashQLA`; local branches `eda-w1` and
`eda-sm100-20261005` point to commit
`423b86cc9a5885e9d8cc6ea476a9422e042eaec8`. Replace
`<FLASHQLA_EDA_REPO_URL>` below with that repository URL after the pinned commit
has been published. The current public main branch does not contain EDA.

```bash
uv pip install --python .venv-lambda/bin/python tilelang==0.1.13
FLASHQLA_PYTHON="$(pwd)/.venv-lambda/bin/python"
git clone '<FLASHQLA_EDA_REPO_URL>' /absolute/path/to/FlashQLA-eda
(
  cd /absolute/path/to/FlashQLA-eda
  git checkout 423b86cc9a5885e9d8cc6ea476a9422e042eaec8
  QLA_VERSION_SUFFIX=+eda.423b86cc9 \
    uv pip install --python "$FLASHQLA_PYTHON" -v .
)
```

This adapts FlashQLA's source installation command (`pip install -v .`) to uv.
The suffix produces distribution version `0.1.3+eda.423b86cc9`; without it,
`setup.py` appends the Git short SHA instead. FlashQLA builds a pure-Python wheel
and does not compile CUDA extensions during installation. TileLang compiles the
kernels at first use. For this qualified SM120 path, provide an RTX 5090, a
compatible driver and a CUDA 13.x development toolkit (including `nvcc`, headers
and libraries), with `CUDA_HOME` pointing to the toolkit and its `bin` on `PATH`.
The general FlashQLA README lists CUDA >=12.8 and Torch >=2.8; the measured
Lambda stack uses CUDA 13.x and TileLang 0.1.13.

Verify the installed distribution and run a small GPU EDA call. The first call
includes JIT compilation; this is a smoke check, not a model-quality evaluation.
`flash_qla.__version__` reports the base version, so check package metadata for
the build suffix:

```bash
.venv-lambda/bin/python - <<'PY'
from importlib.metadata import version

import flash_qla
import torch

assert version("flash_qla") == "0.1.3+eda.423b86cc9"
assert version("tilelang") == "0.1.13"
torch.manual_seed(42)
shape = (1, 16, 1, 128)
q, k, v, e = [torch.randn(shape, device="cuda", dtype=torch.bfloat16)
              for _ in range(4)]
q, k, e = [torch.nn.functional.normalize(x.float(), dim=-1).bfloat16()
           for x in (q, k, e)]
g = -0.1 * torch.rand(shape, device="cuda")
beta = torch.rand(shape[:-1], device="cuda")
gamma = torch.rand_like(beta)
with torch.inference_mode():
    output, state = flash_qla.chunk_eda(
        q, k, v, e, g, beta, gamma, scale=128**-0.5,
        output_final_state=True, backend="tilelang",
    )
assert output.shape == shape
assert state.shape == (1, 1, 128, 128)
assert torch.isfinite(output).all().item()
assert torch.isfinite(state).all().item()
print("FlashQLA EDA GPU smoke check passed")
PY
```

The fork's short convolutions use its own Triton implementation. The
sealed Mesh environment is a read-only numerical oracle, not a serving dependency.

Download the complete compact Lambda export from `chutesai/parallax-8b-lambda`
at revision `1b992902aa52ca487eee21160b45d66791d1e7dc`, export
`exports/242B-tokens_step21067`. Public Hub downloads need no token. Convert and
prepare the execution profile:

```bash
.venv-lambda/bin/python tools/convert_parallax_checkpoint.py \
  --export /absolute/path/to/exports/242B-tokens_step21067 \
  --output /absolute/path/to/lambda-bf16
```

For text serving, copy these three public **Llama-3** tokenizer files into the
converted directory **before** preparing the profile. The export does not record
tokenizer provenance in its manifest, coverage or model config; the sealed
`ops/eval/bench_saves.py` names `NousResearch/Meta-Llama-3-8B`. The following
revision and files were verified without a Hub token:

```bash
.venv-lambda/bin/python - /absolute/path/to/lambda-bf16 <<'PY'
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

target = Path(sys.argv[1])
for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
    source = hf_hub_download(
        "NousResearch/Meta-Llama-3-8B", name,
        revision="315b20096dc791d381d514deb5f8bd9c8d6d3061", token=False,
    )
    shutil.copyfile(source, target / name)
PY

.venv-lambda/bin/python tools/prepare_parallax_profile.py \
  --checkpoint /absolute/path/to/lambda-bf16 \
  --output /absolute/path/to/lambda-sm120
```

The profile symlinks the tokenizer files as well as the weights. If the profile
already exists, copy the three files into that directory too. No tokenizer
weights or Llama model weights are needed. The tokenizer has 128,256 entries
(128,000 base entries); the model has 128,384 rows. A forced padded output ID
128383 was verified to detokenize to empty text without crashing the server.
The tokenizer's BOS is 128000 and EOS is 128001. Mesh benchmark scoring uses
`add_special_tokens=False`; vLLM completions add BOS by default. Use the request
extension `"add_special_tokens": false` to match the mesh convention.

### Verified text serving

From outside the checkout, use one RTX 5090 and the prepared model:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 \
  /absolute/path/to/fork/.venv-lambda/bin/vllm serve \
  /absolute/path/to/lambda-sm120 \
  --host 127.0.0.1 --port 18765 --served-model-name lambda \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.7 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-num-seqs 64 --max-num-batched-tokens 32768 \
  --hf-overrides '{"gdn2_state_storage":"fp32"}'
```

Change the override to `'{"gdn2_state_storage":"fp16"}'` for FP16 recurrent
storage. Despite its inherited name, this field controls EDA state too; dense
weights and attention KV remain BF16. Profile generation defaults to FP32.
Neither `--trust-remote-code` nor a custom load format is required. The tested
upstream automatically falls back to `FULL_DECODE_ONLY` CUDA graphs for this
model, so a compilation override is not required.

```bash
curl --fail-with-body http://127.0.0.1:18765/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lambda","prompt":"The capital of France is","temperature":0,"max_tokens":64,"add_special_tokens":false}'
```

The qualified GPU has 32 GB VRAM. At utilization 0.7, the two state modes reached
**MEASURED sampled peaks of 32,048 / 32,012 MiB** during startup and inference,
and **MEASURED 23,502 / 23,496 MiB** after startup (FP32 / FP16). Expert packing
can briefly fill the card before releasing temporary allocations. Lowering the
utilization setting does not cap that loading peak or qualify a smaller GPU.
At context 8192, **MEASURED utilization 0.24** passed an 8128-token prompt plus
64 outputs in both states; **MEASURED 0.23 failed** cache initialization in FP32.
This is the smallest successful tested setting, with very little cache capacity;
use 0.7 for the tested concurrent workload. Memory samples are taken at 500 ms
intervals and can miss shorter peaks.

The in-process API also initializes the tokenizer. This tested setup uses the
benchmark's explicit graph configuration; the tokenization override matches
the mesh convention:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/absolute/path/to/lambda-sm120",
    dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.7,
    enable_prefix_caching=False, enable_chunked_prefill=False,
    max_num_seqs=64, max_num_batched_tokens=32768,
    compilation_config={
        "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 4, 64],
    },
)
results = llm.generate(
    ["The capital of France is", "def fibonacci(n):",
     "The sun was setting over the quiet village as"],
    SamplingParams(temperature=0, max_tokens=64),
    tokenization_kwargs={"add_special_tokens": False},
)
for result in results:
    print(result.outputs[0].text)
```

This is a base completion model. The France test begins with Paris but repeats;
the story test gives a coherent narrative. The Fibonacci continuation gives
validly formatted recursive Python. These are serving smoke
checks, not a broad language-quality qualification. Both state modes,
64 concurrent requests, CUDA graphs, sparse-FP4 execution and a fresh install
were verified on RTX 5090, 2026-10-05.

Plain serving uses the upstream scheduler and sampler. The tested
`--scheduler-cls vllm.v1.core.sched.parallax_decode.DecodeSlotScheduler` enables
decode-slot caching directly with prefix caching disabled. Selecting
`--worker-extension-cls vllm.v1.worker.gpu.parallax_sampler.PipelineOptimizationWorker`
only exposes worker RPC methods; it does **not** install the in-place sampler.
A local development RPC test verified installation. Lambda's BF16 head recorded
zero reused-logits calls even after installation. Use the benchmark below to
reproduce P4's complete scheduler/sampler setup and throughput settings.

The profile uses a BF16 head and a mixed expert path: packed BF16 below 256
scheduled tokens, sparse ternary FP4 at 256 or more. Batch-256 decode actually
executes the sparse-FP4 CUDA kernels, as verified by profiling. This is not a
pure-FP4 quality qualification at every batch size.
FP32 recurrent storage remains the default. FP16 storage passes the final
multi-step, forced cached-decode NLL condition: its maximum per-case degradation
is **MEASURED +0.007669 nats**, below 0.01 nats, over 8,566 cached positions,
measured on RTX 5090, 2026-10-05.

From outside the checkout, pin one GPU and run the real scheduler/sampler:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 \
  /absolute/path/to/.venv/bin/python \
  /absolute/path/to/fork/benchmarks/kernels/parallax/bench_serving_fp4.py \
  --model /absolute/path/to/lambda-sm120 --output /absolute/path/to/decode.json \
  --state fp32 --engine-context 8192 --batches 1,4,256 --prompts 32,128 \
  --tokens 128 --repeats 5 --warmups 4 --max-batched-tokens 32768 \
  --gpu-memory-utilization 0.7 --temperature 0.8 --top-p 0.95 \
  --synchronize-arrivals --full-decode-graph --capture-sizes 1,4,256 \
  --decode-slot-cache --inplace-sampler
```

Repeat with `--state fp16` for FP16 recurrent storage. For the matched prefill
workload, use `--batches 1,16 --prompts 128,2048 --tokens 1 --repeats 3 --warmups 1`
and retain the other engine/sampling settings. Decode rates exclude prefill and
count 127 cached predictions per request. Prefill rates include first output and
scheduler time. The original Kappa text's claims and reproduction flags are
unchanged.

The generator's default `--eda-optimization auto` selects the qualified `stable`
combination for Lambda. It joins q/k/v/e0/f0/b/c/g0 through the existing
`join_linears` packing, preserving b/c biases; batches matching e1/f1 expansions;
fuses normalization and decay/write/erase gates into the decode recurrence; and
fuses output normalization/gating. Below 256 tokens, the joined GEMM uses explicit
FP32 accumulation and bias before BF16 storage to limit batch-dependent rounding.
At larger batches it uses the existing GEMM path. Decode uses 128-column state
tiles with eight warps for FP32 and four for FP16; batches 1–4 use 32-column tiles
with four warps. These choices come from CUPTI sweeps. Convolution remains separate.
Use `--eda-optimization none` to reproduce the previous Lambda configuration.
Kappa's default profile is unchanged.

The final combination passes the P2b teacher/NLL/self screen and both state
dtypes' exact optimized scheduler shadow checks. The expanded greedy screen uses
all 32 real prompts with 128 generated tokens: **MEASURED 8/32** large-margin first
divergences, against **MEASURED max(F2,F3,F5) = 10/32**. Earlier candidates failed
the gates; no threshold was relaxed. These results were measured on
RTX 5090, 2026-10-05.

Final single-GPU rates below were **MEASURED on RTX 5090, 2026-10-05**,
using the settings above. Decode counts cached output tokens; prefill counts
input tokens.

| Workload | Batch | Prompt | FP32 tok/s | FP16 tok/s |
| --- | --- | --- | --- | --- |
| Decode | 256 | 32 | 23,294 | 28,634 |
| Decode | 256 | 128 | 22,551 | 27,520 |
| Prefill | 16 | 2048 | 68,833 | 69,656 |

For DP8, launch eight independent `bench_serving_fp4.py` processes concurrently,
one per physical GPU with `CUDA_VISIBLE_DEVICES=0` through `7`, TP=1 and separate
output paths. Use the same decode flags above, with `--batches 256`, and repeat
for each state mode. Pass the same fresh directory via `--barrier-dir` to all
processes, `--replica-count 8`, and a unique `--replica-id` from 0 through 7 to
align measured trials. Aggregate cohort decode rate is total cached tokens divided by the span from the earliest first token to the
latest last token across all eight processes. TP greater than one is unqualified.

Final DP8 cohort rates were **MEASURED on RTX 5090, 2026-10-05** with
batch 256 per GPU and the decode settings above:

| Prompt | FP32 cohort tok/s | FP16 cohort tok/s |
| --- | --- | --- |
| 32 | 179,460 | 215,580 |
| 128 | 174,697 | 207,583 |

These are simultaneous cohort measurements.

Forced decode quality reproduction uses
`benchmarks/kernels/parallax/verify_lambda_decode_nll.py --model ... --references
... --window-references ... --state fp32 --output ...`, then the same command with
`--state fp16`. It pre-fills 64 tokens, forces each remaining ground-truth token
with the V2 logits processor, and requests `raw_logprobs`. The first prediction
comes from prefill and is excluded from the cached-decode verdict. Reference
artifacts are built through the sealed `ops.eval.bench_saves.build_model` BF16 lane.
This corpus screen does not establish general language quality.

For diagnostic profiles, use `--eager --component-timing --profile-dir ...` without
scheduler/sampler optimization flags. Component events include diagnostic overhead
and are separate from headline throughput. `bench_eda_decode.py` uses FlashInfer
CUPTI graph timing with cold L2; the measured environment pins `cupti-python==13.2.0`
and `nvidia-cuda-cupti==13.2.75`. The discarded warm-cache fallback is excluded
from the headline rates.
