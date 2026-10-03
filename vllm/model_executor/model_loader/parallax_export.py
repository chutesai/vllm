# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Standalone read-only codecs for canonical compact Kappa exports.

Extracted from mesh's indexer_bundle, relay_model_export, pack_producer and
trunk_client. No fleet/runtime services are imported.
"""

import hashlib
import json
import math
import struct

import regex as re
import torch


class PackError(ValueError):
    pass


class TrunkRefused(ValueError):
    pass


SCHEMA = "indexer_bundle_v1"


class IndexerBundleError(ValueError):
    pass


def parse_header(payload: bytes) -> tuple[dict, int]:
    """Torch-free header parse: (header, body offset)."""
    nl = payload.find(b"\n")
    if nl < 0:
        raise IndexerBundleError("bundle has no header line")
    try:
        header = json.loads(payload[:nl].decode("utf-8"))
    except Exception as exc:
        raise IndexerBundleError(f"bundle header unparsable: {exc}") from exc
    if not isinstance(header, dict):
        raise IndexerBundleError("bundle header must be an object")
    if header.get("schema") != SCHEMA:
        raise IndexerBundleError(f"bundle schema {header.get('schema')!r} != {SCHEMA}")
    return header, nl + 1


_DTYPES = {
    "torch.bfloat16": (torch.bfloat16, 2),
    "torch.float16": (torch.float16, 2),
    "torch.float32": (torch.float32, 4),
}


def unpack_bundle(payload: bytes) -> tuple[dict, dict[str, torch.Tensor]]:
    header, off = parse_header(payload)
    for key in ("term", "step"):
        value = header.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 0xFFFFFFFF
        ):
            raise IndexerBundleError(f"invalid bundle {key}: {value!r}")
    body = payload[off:]
    if hashlib.blake2b(body, digest_size=16).hexdigest() != header.get("checksum"):
        raise IndexerBundleError("bundle checksum mismatch")
    if header.get("dtype") not in _DTYPES:
        raise IndexerBundleError(f"unsupported bundle dtype {header.get('dtype')!r}")
    dtype, itemsize = _DTYPES[header["dtype"]]
    tensors: dict[str, torch.Tensor] = {}
    pos = 0
    entries = header.get("entries")
    if not isinstance(entries, list) or not entries:
        raise IndexerBundleError("bundle entries must be a nonempty list")
    for entry in entries:
        if not isinstance(entry, list) or len(entry) != 3:
            raise IndexerBundleError("malformed bundle entry")
        name, shape, numel = entry
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(shape, list)
            or any(
                isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in shape
            )
            or isinstance(numel, bool)
            or not isinstance(numel, int)
            or numel <= 0
            or math.prod(shape) != numel
        ):
            raise IndexerBundleError("invalid bundle entry geometry")
        if name in tensors:
            raise IndexerBundleError(f"duplicate bundle entry {name}")
        nbytes = int(numel) * itemsize
        raw = body[pos : pos + nbytes]
        if len(raw) != nbytes:
            raise IndexerBundleError(f"bundle truncated at {name}")
        pos += nbytes
        buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        tensors[name] = buf.view(dtype).reshape([int(s) for s in shape]).clone()
    if pos != len(body):
        raise IndexerBundleError("bundle has trailing bytes")
    return header, tensors


STATE_MAGIC = b"MTS1"

_STATE_HEAD = struct.Struct("<4sHI")

STATE_VERSION = 1


def _torch_dtype(name: str) -> torch.dtype:
    if not name.startswith("torch."):
        raise TrunkRefused("composed_layout", f"bad dtype {name!r}")
    dtype = getattr(torch, name[len("torch.") :], None)
    if not isinstance(dtype, torch.dtype):
        raise TrunkRefused("composed_layout", f"bad dtype {name!r}")
    return dtype


def decode_state_blob(blob: bytes) -> tuple[int, dict[str, torch.Tensor]]:
    """Inverse of `encode_state_blob`; the header digest is NOT trusted --
    callers recompute `fragment_digest_of` on the decoded tensors."""
    if len(blob) < _STATE_HEAD.size:
        raise TrunkRefused("composed_layout", "truncated state blob")
    magic, version, hlen = _STATE_HEAD.unpack_from(blob, 0)
    if magic != STATE_MAGIC or version != STATE_VERSION:
        raise TrunkRefused("composed_layout", "not a trunk state blob")
    off = _STATE_HEAD.size
    try:
        header = json.loads(blob[off : off + hlen].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TrunkRefused("composed_layout", f"bad header: {exc}") from exc
    off += hlen
    tensors: dict[str, torch.Tensor] = {}
    for name, dtype_name, shape, nbytes in header["entries"]:
        raw = blob[off : off + int(nbytes)]
        if len(raw) != int(nbytes):
            raise TrunkRefused("composed_layout", f"truncated entry {name!r}")
        off += int(nbytes)
        dtype = _torch_dtype(str(dtype_name))
        shape = tuple(int(s) for s in shape)
        numel = 1
        for s in shape:
            numel *= s
        if numel * torch.empty((), dtype=dtype).element_size() != len(raw):
            raise TrunkRefused("composed_layout", f"entry {name!r} size mismatch")
        buf = bytearray(raw)
        if dtype in (torch.bfloat16, torch.float16):
            t = torch.frombuffer(buf, dtype=torch.int16).view(dtype)
        else:
            t = torch.frombuffer(buf, dtype=dtype)
        tensors[str(name)] = t.reshape(shape).clone()
    if off != len(blob):
        raise TrunkRefused("composed_layout", "trailing bytes in state blob")
    return int(header["fragment_id"]), tensors


FAMILY = "router_avg"


def beta_name(name):
    return name.endswith("router.qb_beta") or name.startswith("qb_beta.")


def wire_dtype_of(layout):
    """The flat dtype of a STORED fragment (pack, export, genesis, hydrate):
    fp32 iff the layout records a beta key in the ``router_avg`` class. Layouts
    that carry no classes (pre-family exports) are bf16. Environment-free, so
    an offline bench process and the trainer read the same bytes the same way."""
    keys = tuple(getattr(layout, "key_order", ()) or ())
    class_of = getattr(layout, "class_of", None)
    if class_of is None:
        return torch.bfloat16
    for key in keys:
        if beta_name(key):
            try:
                if class_of(key) == FAMILY:
                    return torch.float32
            except Exception:  # noqa: BLE001 - a layout without that key's class is a legacy layout
                return torch.bfloat16
    return torch.bfloat16


def _decode_absolute(layout, blob: bytes):
    """Raw bytes of the flat vector in the fragment's wire dtype (bf16, or fp32
    for a quantile router family fragment), or a trunk state blob (MTS1) the
    learner's decoder understands (delegated when available)."""
    import torch

    numel = int(layout.numel)
    if blob[:4] == b"MTS1":
        fid, tensors = decode_state_blob(blob)
        if int(fid) != int(layout.fragment_id):
            raise PackError(
                f"absolute names fragment {fid}, expected {layout.fragment_id}"
            )
        # The entries carry their own dtype: an fp32 entry marks an fp32 fragment
        # (r35 quantile router family); everything else is the bf16 trunk.
        dtype = (
            torch.float32
            if any(t.dtype is torch.float32 for t in tensors.values())
            else torch.bfloat16
        )
        if all(k in tensors for k in layout.key_order):
            parts = [tensors[k].reshape(-1) for k in layout.key_order]
            flat = torch.cat(parts).to(dtype)
        elif len(tensors) == 1:
            (t,) = tensors.values()
            flat = t.detach().reshape(-1).to(dtype)
        else:
            raise PackError("absolute lacks layout keys")
        if int(flat.numel()) != numel:
            raise PackError(f"absolute numel {flat.numel()} != {numel}")
        return flat.clone()
    # Raw bytes: the dtype follows from the length (2 bytes/element bf16, 4 fp32),
    # unambiguous and independent of the process environment or the layout's
    # class annotations; a classed layout must agree with it.
    if len(blob) == numel * 2:
        dtype = torch.bfloat16
    elif len(blob) == numel * 4:
        dtype = torch.float32
    else:
        raise PackError(
            f"absolute has {len(blob)} bytes, expected {numel * 2} (bf16) "
            f"or {numel * 4} (fp32)"
        )
    expected = (
        wire_dtype_of(layout)
        if getattr(layout, "class_of", None) is not None
        else dtype
    )
    if expected is not dtype and getattr(layout, "class_of", None) is not None:
        raise PackError(f"absolute is {dtype} but the layout classes say {expected}")
    if dtype is torch.bfloat16:
        return (
            torch.frombuffer(bytearray(blob), dtype=torch.int16)
            .view(torch.bfloat16)
            .clone()
        )
    return torch.frombuffer(bytearray(blob), dtype=dtype).clone()


_SPLIT_KEY = re.compile(r"(.+)::last\[(\d+):(\d+)\]")


def iter_trunk_groups(manifest, layouts, coverage, source_bytes):
    """Decode every trunk fragment of a relay pack into named model tensors.

    Yields ``(values, provenance)`` groups exactly as ``export_model`` writes
    them: one group per fragment (lane ``trunk``) followed by one group per
    reassembled ``::last[a:b]`` split tensor (lane ``trunk_split``). Layout
    keys are mapped through ``coverage['trunk_names']`` and cast to the
    coverage dtype; a key the coverage does not name refuses. Shared by the
    inference exporter and the trainer's export-init loader so both read the
    same bytes the same way.
    """
    expected = coverage["tensors"]
    split_parts: dict[str, list[tuple[int, int, torch.Tensor, int, int]]] = {}
    for fragment, row in sorted(manifest["trunk"].items(), key=lambda x: int(x[0])):
        layout = layouts[int(fragment)]
        flat = _decode_absolute(layout, source_bytes(row))
        pos, values = 0, {}
        for name, shape in zip(layout.key_order, layout.shapes):
            name = coverage.get("trunk_names", {}).get(name, name)
            n = math.prod(shape)
            tensor = flat[pos : pos + n].reshape(shape).clone()
            match = _SPLIT_KEY.fullmatch(name)
            if match:
                base, start, end = (
                    match.group(1),
                    int(match.group(2)),
                    int(match.group(3)),
                )
                split_parts.setdefault(base, []).append(
                    (start, end, tensor, int(fragment), row["version"])
                )
            else:
                if name not in expected:
                    raise PackError("unexpected trunk tensor: " + name)
                values[name] = tensor.to(
                    getattr(torch, expected[name]["dtype"].removeprefix("torch."))
                )
            pos += n
        if pos != flat.numel():
            raise PackError(
                f"trunk fragment {fragment} layout covers {pos} "
                f"of {flat.numel()} elements"
            )
        yield (
            values,
            {
                "lane": "trunk",
                "fragment": int(fragment),
                "version": row["version"],
                "digest": row["digest"],
            },
        )
    for name, parts in split_parts.items():
        parts.sort(key=lambda part: part[0])
        end = 0
        for start, stop, tensor, _, _ in parts:
            if start != end or stop <= start or tensor.shape[-1] != stop - start:
                raise PackError("split tensor overlap/hole: " + name)
            end = stop
        if name not in expected:
            raise PackError("unexpected split trunk tensor: " + name)
        assembled = torch.cat([p[2] for p in parts], dim=-1).to(
            getattr(torch, expected[name]["dtype"].removeprefix("torch."))
        )
        if list(assembled.shape) != expected[name]["shape"]:
            raise PackError("split tensor incomplete: " + name)
        yield (
            {name: assembled},
            {"lane": "trunk_split", "fragments": [[p[3], p[4]] for p in parts]},
        )


_FACTOR_NAMES = ("index_q_down", "index_q_up", "index_k_down", "index_k_up")

_READINESS_NAMES = (
    "_sparse_readiness_ready",
    "_sparse_readiness_streak",
    "_sparse_readiness_activation_step",
    "_sparse_readiness_recall_ema",
    "_sparse_readiness_mass_ema",
    "_sparse_readiness_new_observations",
)


def derive_inference_state(coverage, header, factors):
    policy = coverage.get("inference_policy")
    if not policy:
        return {}
    if policy.get("schema") != "mesh-inference-policy:1" or policy.get(
        "attention_mode"
    ) not in ("dense", "sparse"):
        raise PackError("unsupported inference policy")
    expected = coverage["tensors"]
    result = {}

    def filled(name, value):
        if name not in expected:
            raise PackError("inference policy names unknown state: " + name)
        spec = expected[name]
        result[name] = torch.full(
            spec["shape"],
            value,
            dtype=getattr(torch, spec["dtype"].removeprefix("torch.")),
        )

    for name in policy["quantile_unused_zero_bias"]:
        filled(name, 0)
    prefixes = policy["indexer_prefixes"]
    expected_factors = {
        prefix + "." + f + ".weight" for prefix in prefixes for f in _FACTOR_NAMES
    }
    if prefixes and (
        header is None or factors is None or set(factors) != expected_factors
    ):
        raise PackError("inference policy requires complete committed indexer bundle")
    for prefix in prefixes:
        for factor in _FACTOR_NAMES:
            name = prefix + "." + factor + "_active"
            result[name] = factors[prefix + "." + factor + ".weight"].clone()
        for suffix, value in (
            ("_indexer_active_step", header["step"]),
            ("_indexer_active_term", header["term"]),
            ("_indexer_active_valid", True),
        ):
            filled(prefix + "." + suffix, value)
        sparse = policy["attention_mode"] == "sparse"
        # These fields are inert in eval except ready, which controls the
        # explicit dense/sparse policy. No invented EMA quality measurement.
        for suffix in _READINESS_NAMES:
            value = (
                sparse
                if suffix == "_sparse_readiness_ready"
                else (-1 if suffix == "_sparse_readiness_activation_step" else 0)
            )
            filled(prefix + "." + suffix, value)
    return result
