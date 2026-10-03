# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify a pinned compact export and materialize a BF16 vLLM baseline."""

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.parallax_export import (
    derive_inference_state,
    iter_trunk_groups,
    unpack_bundle,
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def patterns():
    result = np.zeros((486, 8), dtype=np.int8)
    trits = np.arange(81)[:, None] // (3 ** np.arange(4)) % 3 - 1
    for i, (a, b) in enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))):
        result[i * 81 : (i + 1) * 81, [2 * a, 2 * a + 1, 2 * b, 2 * b + 1]] = trits
    return result


def decode_frame(raw, specs):
    line, body = raw.split(b"\n", 1)
    h = json.loads(line)
    if hashlib.sha256(body).hexdigest() != h["payload_sha256"]:
        raise ValueError("expert checksum mismatch")
    size = (h["numel"] // 8 * 9 + 7) // 8
    if h["schema"] == "compact-expert-pair9:1":
        carry = np.asarray(h["carry"], dtype=np.float32)
        if len(body) != size:
            raise ValueError("bad v1 payload length")
    elif h["schema"] == "compact-expert-pair9:2":
        if h["carry_dtype"] != "bfloat16" or len(body) != size + 2 * h["carry_numel"]:
            raise ValueError("bad v2 carry")
        carry = np.frombuffer(body[size:], dtype=np.uint16) & 0x7FFF
    else:
        raise ValueError("unsupported expert codec")
    if np.any(carry):
        raise ValueError("nonzero low-rank carry requires explicit support")
    data = np.pad(np.frombuffer(body[:size], np.uint8).astype(np.uint32), (0, 2))
    bits = np.arange(h["numel"] // 8, dtype=np.uint32) * 9
    byte = bits >> 3
    code = (
        (data[byte] | data[byte + 1] << 8 | data[byte + 2] << 16) >> (bits & 7)
    ) & 511
    if np.any(code >= 486):
        raise ValueError("invalid pair9 code")
    flat = torch.from_numpy(patterns()[code].reshape(-1))
    alphas = torch.tensor(h["alphas"], dtype=torch.float32)
    scales = h.get("scale_rows")
    if scales is not None and (
        h.get("scale_format") != 1
        or list(scales) != [s["shape"][0] for s in specs]
        or alphas.numel() != sum(scales)
    ):
        raise ValueError("unsupported row scale geometry")
    offset = 0
    result = {}
    for s in specs:
        n = math.prod(s["shape"])
        codes = flat[offset : offset + n].reshape(s["shape"])
        offset += n
        if scales is None:
            alpha = alphas[s["alpha_index"]]
        else:
            row = 0 if s["alpha_index"] == 0 else scales[0]
            alpha = alphas[row : row + s["shape"][0], None]
        result[s["name"]] = (codes.float() * alpha).to(torch.bfloat16).contiguous()
    if offset != flat.numel() or not all(
        torch.isfinite(t).all() for t in result.values()
    ):
        raise ValueError("invalid expert tensors")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.export.resolve()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / "manifest.json").read_text())
    coverage = json.loads((root / "coverage.json").read_text())
    config = json.loads((root / "model_config.json").read_text())
    pack = json.loads((root / "relay_pack/manifest.json").read_text())
    if (
        meta["schema"] != "mesh-relay-compact-export:1"
        or coverage["ternary24_contract"] != "pair8"
    ):
        raise ValueError("unsupported export")
    with tarfile.open(root / "packed_experts.tar") as archive:
        archive.extractall(root, filter="data")
    for relative, spec in meta["payload_files"].items():
        path = root / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or path.stat().st_size != spec["nbytes"]
            or digest(path) != spec["sha256"]
        ):
            raise ValueError("payload verification failed: " + relative)
    print("PAYLOADS_VERIFIED", flush=True)
    layouts = {
        int(k): SimpleNamespace(**v)
        for k, v in json.loads((root / "layouts.json").read_text()).items()
    }
    seen = set()
    index = {}
    shard = 0

    def write(values):
        nonlocal shard
        checked = {}
        for name, tensor in values.items():
            spec = meta["tensors"][name]
            raw = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            if hashlib.sha256(raw).hexdigest() != spec["tensor_sha256"]:
                raise ValueError("canonical tensor hash mismatch: " + name)
            if name in seen:
                continue
            seen.add(name)
            checked[name] = tensor.contiguous()
        if checked:
            filename = f"model-{shard:05d}.safetensors"
            shard += 1
            save_file(checked, str(out / filename))
            index.update({name: filename for name in checked})

    for values, _ in iter_trunk_groups(
        pack,
        layouts,
        coverage,
        lambda row: (root / "relay_pack" / row["path"]).read_bytes(),
    ):
        write(values)
    header, factors = unpack_bundle((root / "indexer.bundle").read_bytes())
    write(factors)
    write(derive_inference_state(coverage, header, factors))
    with tarfile.open(root / "packed_experts.tar") as archive:
        for j, (key, specs) in enumerate(coverage["experts"].items()):
            relative = meta["packed_experts"][pack["experts"][key]["path"]]
            member = archive.getmember(relative)
            if not member.isfile():
                raise ValueError("expert archive entry not a file")
            with archive.extractfile(member) as stream:
                write(decode_frame(stream.read(), specs))
            if j % 128 == 0:
                print("EXPERT", j, flush=True)
    missing = set(meta["tensors"]) - seen - set(coverage.get("tied_tensors", {}))
    if missing:
        raise ValueError("uncovered tensors: " + str(sorted(missing)))
    for source in coverage.get("tied_tensors", {}).values():
        if source not in index:
            raise ValueError("missing tied source")
    config.update(
        model_type="parallax",
        architectures=["ParallaxForCausalLM"],
        random_weights=False,
        expert_backend="bf16",
        dense_precision="bf16",
        checkpoint_format="kappa_bf16",
        max_position_embeddings=config["max_seq_len"],
        logit_scale_max=coverage["inference_policy"].get("logit_scale_max"),
        inference_attention_mode=coverage["inference_policy"]["attention_mode"],
    )
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index}, indent=2)
    )
    (out / "verification.json").write_text(
        json.dumps(
            {
                "source_manifest_sha256": digest(root / "manifest.json"),
                "verified_tensors": len(seen),
                "shards": shard,
                "tied_tensors": coverage.get("tied_tensors", {}),
            },
            indent=2,
        )
    )
    print("CONVERSION_COMPLETE", len(seen), flush=True)


if __name__ == "__main__":
    main()
