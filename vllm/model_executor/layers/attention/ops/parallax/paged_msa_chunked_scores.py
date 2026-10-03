# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact block-max index selection and causal attention over vLLM KV pages.

The index branch has four query heads and one key head. Main Q/K/V are 16/4/4.
Selection is independent per KV group, includes the current 128-token block,
and never attends to a future token or another request's page.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _scores(
    Q,
    K,
    TABLE,
    STARTS,
    LENS,
    SCORES,
    POS,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCKS: tl.constexpr,
    SEQS: tl.int32,
    BS: tl.constexpr,
):
    token, group, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < SEQS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), axis=0)
    if seq < SEQS:
        begin, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
        pos = tl.load(LENS + seq) - (end - begin) + token - begin
        if group == 0 and block == 0:
            tl.store(POS + token, pos)
        ds = tl.arange(0, 128)
        loc = block * 128 + tl.arange(0, 128)
        q = tl.load(Q + (token * GROUPS + group) * 128 + ds).to(tl.float32)
        page = tl.load(TABLE + seq * TS + loc // PAGE, loc <= pos, other=0)
        k = tl.load(
            K + page[:, None] * KS0 + (loc[:, None] % PAGE) * KS1 + ds[None, :],
            loc[:, None] <= pos,
            other=0,
        ).to(tl.float32)
        dot = tl.sum(k * q[None, :], axis=1) * (128**-0.5)
        score = tl.max(tl.where(loc <= pos, dot, -float("inf")), axis=0)
        tl.store(SCORES + (token * GROUPS + group) * BLOCKS + block, score)
    else:
        tl.store(SCORES + (token * GROUPS + group) * BLOCKS + block, -float("inf"))
        if group == 0 and block == 0:
            tl.store(POS + token, -1)


@triton.jit
def _scores_chunked(
    Q,
    K,
    TABLE,
    STARTS,
    LENS,
    SCORES,
    POS,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCKS: tl.constexpr,
    SEQS: tl.int32,
    BS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    token, group, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    first = chunk * CHUNK
    cols = first + tl.arange(0, CHUNK)
    row = (token * GROUPS + group) * BLOCKS
    tl.store(SCORES + row + cols, -float("inf"), cols < BLOCKS)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < SEQS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), axis=0)
    if seq < SEQS:
        begin, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
        pos = tl.load(LENS + seq) - (end - begin) + token - begin
        if group == 0 and chunk == 0:
            tl.store(POS + token, pos)
        if first * 128 <= pos:
            ds = tl.arange(0, 128)
            q = tl.load(Q + (token * GROUPS + group) * 128 + ds).to(tl.float32)
            stop = tl.minimum(tl.minimum(first + CHUNK, BLOCKS), (pos + 128) // 128)
            for block in range(first, stop):
                loc = block * 128 + tl.arange(0, 128)
                page = tl.load(TABLE + seq * TS + loc // PAGE, loc <= pos, other=0)
                k = tl.load(
                    K + page[:, None] * KS0 + (loc[:, None] % PAGE) * KS1 + ds[None, :],
                    loc[:, None] <= pos,
                    other=0,
                ).to(tl.float32)
                dot = tl.sum(k * q[None, :], axis=1) * (128**-0.5)
                score = tl.max(tl.where(loc <= pos, dot, -float("inf")), axis=0)
                tl.store(SCORES + row + block, score)
    elif group == 0 and chunk == 0:
        tl.store(POS + token, -1)


@triton.jit
def _pad_scores(SCORES, POS, STARTS, T: tl.int32, COLS: tl.constexpr, SEQS: tl.int32):
    actual = tl.load(STARTS + SEQS)
    row = tl.program_id(0) * 32 + tl.arange(0, 32)
    if tl.program_id(0) * 32 + 31 >= actual:
        cols = tl.program_id(1) * 128 + tl.arange(0, 128)
        mask = (row >= actual) & (row < T)
        tl.store(
            SCORES + row[:, None] * COLS + cols[None, :],
            -float("inf"),
            mask[:, None] & (cols[None, :] < COLS),
        )
        if tl.program_id(1) == 0:
            tl.store(POS + row, -1, mask)


@triton.jit
def _scores_prefill(
    Q,
    K,
    TABLE,
    STARTS,
    LENS,
    SCORES,
    POS,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCKS: tl.constexpr,
    SEQS: tl.int32,
    TOKENS: tl.int32,
    QR: tl.constexpr,
    NB: tl.constexpr = 1,
):
    seq = tl.program_id(1)
    row = tl.arange(0, QR)
    group = row % GROUPS
    query_tile = QR // GROUPS
    block = tl.program_id(2) * NB
    begin, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
    token = begin + tl.program_id(0) * query_tile + row // GROUPS
    write = (token < end) | ((seq == SEQS - 1) & (token < TOKENS))
    if begin + tl.program_id(0) * query_tile < end or (
        seq == SEQS - 1 and begin + tl.program_id(0) * query_tile < TOKENS
    ):
        length = tl.load(LENS + seq)
        pos = length - (end - begin) + token - begin
        valid = (token < end) & (token < TOKENS)
        if block == 0:
            tl.store(POS + token, tl.where(valid, pos, -1), write & (group == 0))
        score = tl.full((QR, NB), -float("inf"), tl.float32)
        if (
            block * 128 < length
            and block * 128
            <= length - (end - begin) + tl.program_id(0) * query_tile + query_tile - 1
        ):
            KD: tl.constexpr = 128 if NB == 1 else 64
            cols = tl.arange(0, 128 * NB)
            loc = block * 128 + cols
            page = tl.load(TABLE + seq * TS + loc // PAGE, loc < length, other=0)
            dots = tl.full((QR, 128 * NB), 0, tl.float32)
            for piece in range(128 // KD):
                ds = piece * KD + tl.arange(0, KD)
                q = tl.load(
                    Q + (token[:, None] * GROUPS + group[:, None]) * 128 + ds[None, :],
                    valid[:, None],
                    other=0,
                )
                key = tl.load(
                    K + page[None, :] * KS0 + (loc[None, :] % PAGE) * KS1 + ds[:, None],
                    loc[None, :] < length,
                    other=0,
                )
                dots = tl.dot(q, key, dots, input_precision="tf32x3")
            dots *= 128**-0.5
            dots = tl.where(
                valid[:, None] & (loc[None, :] <= pos[:, None]), dots, -float("inf")
            )
            score = tl.max(dots.reshape(QR, NB, 128), axis=2)
        block_ids = block + tl.arange(0, NB)
        tl.store(
            SCORES + (token * GROUPS + group)[:, None] * BLOCKS + block_ids[None, :],
            score,
            write[:, None] & (block_ids[None, :] < BLOCKS),
        )


@triton.jit
def _attend(
    Q,
    K,
    V,
    TABLE,
    STARTS,
    LENS,
    SELECTED,
    OUT,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    TS: tl.constexpr,
    PAGE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    TOP: tl.constexpr,
    SEQS: tl.int32,
    BS: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    group = head // (HQ // HK)
    ss = tl.arange(0, BS)
    ends = tl.load(STARTS + ss + 1, ss < SEQS, other=2147483647)
    seq = tl.sum((token >= ends).to(tl.int32), axis=0)
    if seq < SEQS:
        begin, end = tl.load(STARTS + seq), tl.load(STARTS + seq + 1)
        pos = tl.load(LENS + seq) - (end - begin) + token - begin
        ds = tl.arange(0, 128)
        ns = tl.arange(0, 128)
        q = tl.load(Q + (token * HQ + head) * 128 + ds).to(tl.float32)
        acc = tl.full((128,), 0, tl.float32)
        maximum, denom = -float("inf"), 0.0
        for b in range(TOP):
            block = tl.load(SELECTED + (token * HK + group) * TOP + b)
            if block >= 0:
                loc = block * 128 + ns
                page = tl.load(TABLE + seq * TS + loc // PAGE, loc <= pos, other=0)
                off = page[:, None] * KS0 + (loc[:, None] % PAGE) * KS1
                off += group * KS2 + ds[None, :]
                k = tl.load(K + off, loc[:, None] <= pos, other=0).to(tl.float32)
                v = tl.load(V + off, loc[:, None] <= pos, other=0).to(tl.float32)
                scores = tl.sum(k * q[None, :], axis=1) * (128**-0.5)
                scores = tl.where(loc <= pos, scores, -float("inf"))
                new_max = tl.maximum(maximum, tl.max(scores, axis=0))
                alpha = tl.exp(maximum - new_max)
                p = tl.exp(scores - new_max)
                acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
                denom = denom * alpha + tl.sum(p, axis=0)
                maximum = new_max
        tl.store(OUT + (token * HQ + head) * 128 + ds, acc / denom)
    else:
        ds = tl.arange(0, 128)
        tl.store(OUT + (token * HQ + head) * 128 + ds, 0.0)


def paged_msa(
    q,
    qi,
    k,
    v,
    ki,
    table,
    index_table,
    starts,
    lengths,
    max_seq_len,
    topk=16,
    tensorcore_scores=False,
    tensorcore_attend=False,
    max_query_len=None,
    score_tile=16,
    score_blocks=1,
    direct_attend=False,
    bounded_attend=False,
    latent_projection=None,
    latent_hot=None,
    latent_keys=None,
    split_decode=False,
    fp32_probabilities=False,
    grouped_scalar=False,
    grouped_scalar_maxnreg=None,
    grouped_scalar_heads=4,
    dense_prefix_context=None,
    dense_probability_components=0,
    dense_probability_dtype="bf16",
    dense_query_tile=32,
    dense_warps=4,
    late_v_decode=False,
    fused_selection=False,
):
    """Cache views are [physical pages, tokens/page, heads, 128]."""
    tokens, hq, dim = q.shape
    hk = qi.shape[1] if latent_projection is not None else k.shape[2]
    assert dim == 128 and qi.shape == (tokens, hk, 128)
    assert k.stride() == v.stride() and k.stride(-1) == 1
    assert ki.shape[2] == 1 and ki.stride(-1) == 1
    assert q.is_contiguous() and qi.is_contiguous()
    if dense_prefix_context is not None:
        if (
            latent_projection is not None
            or dense_prefix_context > topk * 128
            or max_query_len is None
        ):
            raise ValueError(
                "Dense FP32 prefill requires expanded KV and all blocks selected"
            )
        if dense_probability_dtype == "fp16":
            from .dense_fp16_attention import (
                dense_fp32_attention,
            )

            return dense_fp32_attention(
                q,
                k,
                v,
                starts,
                lengths,
                table,
                max_query_len,
                dense_prefix_context,
                components=dense_probability_components,
                query_tile=dense_query_tile,
                warps=dense_warps,
            ), None
        if dense_probability_dtype != "bf16":
            raise ValueError("Unsupported dense PV precision")
        if (dense_probability_components, dense_query_tile, dense_warps) != (0, 32, 4):
            from .dense_bf16_residual_attention import (
                dense_fp32_attention,
            )

            return dense_fp32_attention(
                q,
                k,
                v,
                starts,
                lengths,
                table,
                max_query_len,
                dense_prefix_context,
                components=dense_probability_components,
                query_tile=dense_query_tile,
                warps=dense_warps,
            ), None
        from .dense_fp32_attention import dense_fp32_attention as dense_scalar_attention

        return dense_scalar_attention(
            q, k, v, starts, lengths, table, max_query_len, dense_prefix_context
        ), None
    seqs = lengths.numel()
    blocks = triton.cdiv(max_seq_len, 128)
    scores = torch.empty((tokens, hk, blocks), device=q.device, dtype=torch.float32)
    positions = torch.empty(tokens, device=q.device, dtype=torch.int32)
    if tensorcore_scores:
        if score_blocks not in (1, 2, 4):
            raise ValueError("Index block group must be 1, 2 or 4")
        if score_tile not in (16, 32, 64, 128):
            raise ValueError("Index query tile must be 16, 32, 64 or 128")
        # A CPU metadata bound, never a GPU scalar read or synchronization.
        query_bound = tokens if max_query_len is None else max_query_len
        if hk not in (1, 2, 4, 8, 16):
            raise ValueError("Grouped index scoring needs a head count dividing 16")
        _pad_scores[(triton.cdiv(tokens, 32), triton.cdiv(hk * blocks, 128))](
            scores, positions, starts, tokens, hk * blocks, seqs, num_warps=4
        )
        _scores_prefill[
            (
                triton.cdiv(query_bound, score_tile // hk),
                seqs,
                triton.cdiv(blocks, score_blocks),
            )
        ](
            qi,
            ki,
            index_table,
            starts,
            lengths,
            scores,
            positions,
            *ki.stride()[:3],
            index_table.stride(0),
            ki.shape[1],
            hk,
            blocks,
            seqs,
            tokens,
            score_tile,
            score_blocks,
            num_warps=4,
            num_stages=1 if score_blocks > 1 else 3,
        )
    else:
        score_kernel = _scores_chunked if max_query_len == 1 else _scores
        chunk = 4 if max_query_len == 1 else 1
        extra = {"CHUNK": chunk} if max_query_len == 1 else {}
        score_kernel[(tokens, hk, triton.cdiv(blocks, chunk))](
            qi,
            ki,
            index_table,
            starts,
            lengths,
            scores,
            positions,
            *ki.stride()[:3],
            index_table.stride(0),
            ki.shape[1],
            hk,
            blocks,
            seqs,
            triton.next_power_of_2(seqs),
            num_warps=4,
            **extra,
        )
    vals, selected = scores.topk(min(topk, blocks), dim=-1)
    if fused_selection:
        from .msa_selection_finalize import (
            finalize_selection,
        )

        selected = finalize_selection(vals, selected, positions)
    else:
        selected = torch.where(vals.isfinite(), selected, -1)
        local = (positions // 128)[:, None, None]
        present = (selected == local).any(dim=-1, keepdim=True)
        selected[..., -1:] = torch.where(present, selected[..., -1:], local)
        selected = selected.to(torch.int32).contiguous()
    if grouped_scalar:
        if latent_projection is not None:
            raise ValueError("Grouped scalar attention requires expanded KV")
        if grouped_scalar_heads == 4:
            from .grouped_scalar_attention import (
                grouped_scalar_attention,
            )
        else:
            from .grouped_scalar_heads import (
                grouped_scalar_attention,
            )

        return grouped_scalar_attention(
            q,
            k,
            v,
            selected,
            positions,
            starts,
            table,
            maxnreg=grouped_scalar_maxnreg,
            **(
                {"heads_per_program": grouped_scalar_heads}
                if grouped_scalar_heads != 4
                else {}
            ),
        ), selected
    if latent_projection is not None:
        wk, wv, rope = latent_projection
        if latent_keys is not None:
            from .key_latent_attention import (
                key_latent_attention,
            )

            out = key_latent_attention(
                q,
                latent_keys,
                k,
                wv,
                selected,
                positions,
                starts,
                table,
                prefill=max_query_len is not None and max_query_len >= 4,
            )
        elif max_query_len is not None and max_query_len >= 4:
            from .latent_cached_attention import (
                latent_cached_attention,
            )

            out = latent_cached_attention(
                q,
                k,
                wk,
                wv,
                rope,
                selected,
                positions,
                starts,
                table,
                max_context=max_seq_len,
            )
        else:
            from .latent_decode import (
                latent_decode,
            )

            hot = latent_hot if tokens >= 4 else None
            if hot is not None:
                from .hot_keys import (
                    prepare_hot_keys,
                )

                prepare_hot_keys(
                    q, k, wk, rope, selected, positions, starts, table, hot
                )
            # Offline SM120 tuning; no runtime autotune or GPU->CPU read.
            out = latent_decode(
                q,
                k,
                wk,
                wv,
                rope,
                selected,
                positions,
                starts,
                table,
                block_tokens=64 if tokens < 4 else 32,
                splits=8 if 4 <= tokens < 32 else 16,
                num_warps=8 if tokens < 4 else 4,
                hot=hot,
            )
        return out, selected
    if split_decode and max_query_len == 1 and tokens < 128:
        from .split_selected_attention import (
            split_selected_attention,
        )

        return split_selected_attention(
            q, k, v, selected, positions, starts, table
        ), selected
    if tensorcore_attend:
        if direct_attend:
            from .direct_selected_attention import (
                direct_selected_attention,
            )

            out = direct_selected_attention(
                q,
                k,
                v,
                selected,
                positions,
                starts,
                table,
                bounded_loop=bounded_attend,
                fp32_probabilities=fp32_probabilities,
                block_tokens=64
                if max_query_len is not None and max_query_len >= 128
                else 128,
            )
            return out, selected
        else:
            from .selected_attention import (
                selected_attention,
            )

        out = selected_attention(q, k, v, selected, positions, starts, table)
        return out, selected
    out = torch.empty_like(q)
    attend_impl = _attend
    if late_v_decode and max_query_len == 1:
        from .msa_late_v import (
            _attend_late_v as attend_impl,
        )
    attend_impl[(tokens, hq)](
        q,
        k,
        v,
        table,
        starts,
        lengths,
        selected,
        out,
        *k.stride()[:3],
        table.stride(0),
        k.shape[1],
        hq,
        hk,
        selected.shape[-1],
        seqs,
        triton.next_power_of_2(seqs),
        num_warps=8,
    )
    return out, selected
