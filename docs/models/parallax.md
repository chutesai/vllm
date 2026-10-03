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
