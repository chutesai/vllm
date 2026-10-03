# Native Parallax integration validation

Upstream base: `5f30fc7031cae49bf51073fc953d419b08f8887c`.
Checkpoint: `chutesai/parallax-8b-kappa@15975e88320522284396bfe15c02e533d8b49dc4`,
step 37898. Device: one RTX 5090. Torch 2.13.0+cu132; Triton 3.7.1;
Python 3.12. Runtime version: `0.30.1rc1.dev619+g5f30fc703.parallax`.

The model/config registries, checkpoint loader, Python kernels and compiled
SM120a extension run from the native fork. An import guard rejects both `mesh`
and `parallax_inference` in candidate processes. The plugin control runs allow
the old plugin only to establish the reference.

## Numerical and throughput results

- `native-integration-parity.json`: 256 sequences, 128 prompt tokens and 128
  generated tokens each; all 32,768 generated tokens and logprobs exactly match
  the prior reference, maximum logprob delta zero. All 32,000 predicted cache
  no-ops passed shadow checking against upstream allocation.
- `native-integration-serving.json`: concurrency 256, synchronized arrivals,
  temperature 0.8, top-p 0.95, 128 output tokens, four warmups and five trials.
  Median aggregate decode-only rates are 23,853 tokens/s at prompt length 32 and
  22,810 tokens/s at prompt length 128. All five sampled-token hashes for each
  shape match `../kappa_upstream_5f30_20261003/fresh-upstream-serving.json`.
- `native-integration-prefill.json`: at concurrency 16 and prompt length 2,048,
  67,946 input tokens/s including the first output and scheduler time. Three
  trials after one warmup. The receipt also contains smaller prefill shapes.
- `native-wheel-small-parity.json`: installed-wheel inference matches the prior
  plugin at batch sizes 1 and 4, prompt length 128 and 32 generated tokens. The
  isolated candidate process exits successfully. Single-trial rates are a smoke
  check, not a performance ranking.
- Focused tests: nine config/export/profile tests, six sampler tests and one
  cache-contract test pass. Targeted Ruff and `git diff --check` pass locally.

These are synthetic token-ID workloads. Exact port comparisons use the same
approximate single-component FP4 expert activation profile; they do not prove
equivalence to BF16 expert activations or establish language-model quality.
Dense trunk/KV use BF16, routing and recurrent state use FP32. Small-batch
execution uses the pair-LUT path. No multi-GPU, prefix-sharing or speculative
decoding qualification is implied.

## Build coverage

Installation reuses upstream CUDA 13 binaries pinned to the base commit and
compiles the fork's `_parallax_C` through its normal CMake extension build.
`native-integration-wheel-smoke.json` records the successful installed model/config
registry and native-extension imports outside the checkout with both old packages
blocked. `native-integration-wheel-sources.json` verifies all 52 recorded runtime
and config files in that wheel match the local fork.

The extension uses the CPython ABI, so its wheel is tagged `cp312-cp312` rather
than claiming the stable ABI. Full upstream CUDA source compilation and broad
upstream CI have not been run.

`integration-map.json` records the old-to-native module relocation and retired
experimental options. `source-sha256.json` identifies the source/config/build
files in this validation.

## Upstream lint pipeline

`lint-validation.json` records the successful complete Linux run of CI's
`pre-commit run --all-files --hook-stage manual`. All configured manual-stage
hooks passed, including Python 3.10–3.13 mypy, Ruff check/format, clang-format,
typos, Markdown, workflow/shell checks, dependency locks, SPDX, forbidden imports,
config validation and test-to-CI coverage. No hooks were skipped or weakened.

The run used an isolated indexed snapshot containing all new files; the actual
fork remains unstaged and uncommitted. The root development container's child
process dropped DAC override/read-search capabilities so the upstream read-only
HOME fixture behaves as it does under ordinary filesystem permissions. Earlier
macOS and root-specific failures are recorded in the receipt.

Standards fixes use the native Triton/TileLang wrappers and Transformers config
name, add required headers and type declarations, and remove unreachable remnants.
`post-lint-source-sha256.json` identifies the current source; the original source
manifest remains as the initial integration receipt. `post-lint-parity.json`
still matches all 32,768 tokens and logprobs exactly. The rebuilt wheel passes
all 16 focused tests, import/source checks and batch-1/batch-4 LUT token parity;
its receipts are `post-lint-wheel-*.json`.
