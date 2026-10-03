# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit canonical-export mapping; unknown state and omissions refuse."""

import math

import regex as re
import torch


def load_kappa_weights(model, weights):
    if model.config.random_weights:
        raise ValueError("Use the explicit random loader for random weights")
    targets = dict(model.named_parameters()) | dict(model.named_buffers())
    loaded = set()
    pending = {}
    seen = set()

    def copy(name, value):
        if name not in targets:
            raise ValueError("Unknown runtime target: " + name)
        target = targets[name]
        if target.shape != value.shape:
            raise ValueError(f"Wrong geometry: {name}: {value.shape} != {target.shape}")
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint: " + name)
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))
        loaded.add(name)

    for name, value in weights:
        if name in seen:
            raise ValueError("Duplicate checkpoint tensor: " + name)
        seen.add(name)
        if name == "final_norm.logit_scale_log":
            limit = getattr(model.config, "logit_scale_max", None)
            if limit is not None:
                value = value.float().clamp(max=math.log(limit))
            copy("logit_scale", value.float().exp())
            continue
        if name == "final_norm.weight" and model.config.final_norm_unit_gain:
            value = value.to(targets[name].dtype)
            value = (value.float() / value.float().square().mean().add(1e-6).sqrt()).to(
                value.dtype
            )
        if re.fullmatch(r"layers\.\d+\._(?:indexer|sparse_readiness)_.*", name):
            continue  # Export policy and active factors determine inference behavior.
        match = re.fullmatch(
            r"layers\.(\d+)\.index_([qk])_(down|up)(?:\.weight|_active)", name
        )
        if match:
            i, qk, direction = match.groups()
            if name.endswith("_active"):
                copy(
                    f"layers.{i}.index_{qk}.{0 if direction == 'down' else 1}.weight",
                    value,
                )
            # Converter validates shadow factors; selection uses active factors.
            continue
        match = re.fullmatch(
            r"layers\.(\d+)\.experts\.(\d+)\.(up|down)_proj\.weight", name
        )
        if match:
            i, e, direction = match.groups()
            bank = getattr(model.layers[int(i)], direction)
            if bank[int(e)].shape != value.shape:
                raise ValueError("Expert shape mismatch")
            bank[int(e)].copy_(value.to(device=bank.device, dtype=bank.dtype))
            loaded.add(f"layers.{i}.{direction}")
            continue
        name = name.replace(".router.gate.weight", ".router.weight").replace(
            ".router.qb_beta", ".qb_beta"
        )
        if name.endswith(".router.expert_bias"):
            if torch.count_nonzero(value):
                raise ValueError("Nonzero quantile expert bias")
            continue
        name = name.replace(".k_up_proj.", ".k_proj.").replace(
            ".v_up_proj.", ".v_proj."
        )
        match = re.fullmatch(r"layers\.(\d+)\._fast_impl\.(.+)", name)
        if match:
            i, suffix = match.groups()
            if suffix in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "b_proj.weight",
                "w_proj.weight",
                "q_conv1d.weight",
                "k_conv1d.weight",
                "v_conv1d.weight",
            ):
                pending[(i, suffix)] = value.clone()
                continue
            name = f"layers.{i}.{suffix}"
        name = re.sub(
            r"\.shared_experts\.(\d+)\.up_proj\.", r".shared_experts.\1.0.", name
        )
        name = re.sub(
            r"\.shared_experts\.(\d+)\.down_proj\.", r".shared_experts.\1.2.", name
        )
        if name == "lm_head.weight" and model.config.tie_word_embeddings:
            raise ValueError("Tied head must have one canonical embedding source")
        copy(name, value)
    for i, kind in enumerate(model.config.hybrid_override_pattern):
        if kind != "R":
            continue

        def concat(parts, layer_index=i):
            return torch.cat([pending.pop((str(layer_index), p)) for p in parts], dim=0)

        copy(
            f"layers.{i}.qkv.weight",
            concat(["q_proj.weight", "k_proj.weight", "v_proj.weight"]),
        )
        copy(f"layers.{i}.bw.weight", concat(["b_proj.weight", "w_proj.weight"]))
        copy(
            f"layers.{i}.conv_weight",
            concat(["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"]).squeeze(
                1
            ),
        )
    if pending:
        raise ValueError("Unconsumed GDN tensors")
    expected_experts = {
        f"layers.{i}.experts.{e}.{direction}_proj.weight"
        for i, kind in enumerate(model.config.hybrid_override_pattern)
        if kind == "E"
        for e in range(model.config.n_routed_experts)
        for direction in ("up", "down")
    }
    if not expected_experts.issubset(seen):
        raise ValueError("Missing individual expert tensors")
    required = set(dict(model.named_parameters())) | {"logit_scale"}
    for i, kind in enumerate(model.config.hybrid_override_pattern):
        if kind == "E":
            required.update(
                {f"layers.{i}.up", f"layers.{i}.down", f"layers.{i}.qb_beta"}
            )
    missing = required - loaded
    if missing:
        raise ValueError("Missing runtime weights: " + str(sorted(missing)))
    for layer_index, layer in enumerate(model.layers):
        if getattr(layer, "backend", None) != "packed_bf16":
            continue
        from ..layers.fused_moe.parallax.ternary_decode import pack

        for direction in ("up", "down"):
            dense = getattr(layer, direction)
            scale = dense.abs().amax(-1).contiguous()
            codes = dense.sign().to(torch.int8)
            if not torch.equal(dense, codes.to(dense.dtype) * scale[..., None]):
                raise ValueError("Expert weights are not losslessly row-scaled ternary")
            if getattr(layer, "native_sparse_bf16", False) or getattr(
                layer, "pair_lut_decode", False
            ):
                from ..layers.fused_moe.parallax.packed_sparse_bf16 import (
                    pack_sparse,
                )

                sparse, meta = pack_sparse(codes)
                layer.register_buffer("sparse_" + direction, sparse)
                layer.register_buffer("meta_" + direction, meta)
                if getattr(layer, "expanded_sparse_prefill", False):
                    from ..layers.fused_moe.parallax.packed_sparse_bf16 import (
                        expand_sparse,
                    )

                    layer.register_buffer(
                        "expanded_sparse_" + direction, expand_sparse(sparse, scale)
                    )
            setattr(layer, direction, pack(codes))
            setattr(layer, "alpha_" + direction, scale)
            targets.pop(f"layers.{layer_index}.{direction}", None)
            del dense, codes
    if getattr(model.config, "joined_bf16_gdn2", False):
        from ..layers.parallax_linear import join_linears

        for i, kind in enumerate(model.config.hybrid_override_pattern):
            if kind == "R":
                layer = model.layers[i]
                layer.joined_input = join_linears(
                    (layer.qkv, layer.bw, layer.f_proj[0], layer.g_proj[0])
                )
                loaded.add(f"layers.{i}.joined_input.weight")
    if getattr(model.config, "single_sparse_fp4_components", 0):
        from ..layers.fused_moe.parallax.sparse_ternary_fp4 import (
            SparseTernaryFP4Experts,
        )

        def unpack(w):
            shifts = torch.arange(4, device=w.device, dtype=torch.uint8) * 2
            code = ((w[..., None] >> shifts) & 3).flatten(-2).to(torch.int8)
            return ((1 - (code & 2)) * (code != 0).to(torch.int8)).contiguous()

        for layer in model.layers:
            if getattr(layer, "backend", None) == "packed_bf16":
                layer.single_sparse_fp4 = SparseTernaryFP4Experts(
                    unpack(layer.up),
                    unpack(layer.down),
                    layer.alpha_up,
                    layer.alpha_down,
                )
    return loaded
