# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One ordered write of compact latent and index state to distinct paged pools."""

from vllm.triton_utils import tl, triton


@triton.jit
def _store(
    C,
    INDEX_INPUT,
    CSLOTS,
    ISLOTS,
    CCACHE,
    ICACHE,
    TAGS,
    HOT: tl.constexpr,
    CAP: tl.constexpr,
    N: tl.constexpr,
    CNS: tl.constexpr,
    INS: tl.constexpr,
    CS: tl.constexpr,
    IS: tl.constexpr,
    CP: tl.constexpr,
    IP: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    d = tl.arange(0, 256)
    cs = tl.load(CSLOTS + row, (row < N) & (row < CNS), other=-1)
    ix = tl.load(ISLOTS + row, (row < N) & (row < INS), other=-1)
    c = tl.load(
        C + row[:, None] * CS + d[None, :],
        (row[:, None] < N) & (cs[:, None] >= 0),
        other=0,
    )
    i = tl.load(
        INDEX_INPUT + row[:, None] * IS + d[None, :],
        (row[:, None] < N) & (ix[:, None] >= 0) & (d[None, :] < 128),
        other=0,
    )
    tl.store(
        CCACHE + (cs[:, None] // CP) * C0 + (cs[:, None] % CP) * C1 + d[None, :],
        c,
        (row[:, None] < N) & (cs[:, None] >= 0),
    )
    tl.store(
        ICACHE + (ix[:, None] // IP) * I0 + (ix[:, None] % IP) * I1 + d[None, :],
        i,
        (row[:, None] < N) & (ix[:, None] >= 0) & (d[None, :] < 128),
    )

    if HOT:
        tl.atomic_xchg(
            TAGS + (cs.to(tl.int64) // 16) % CAP,
            -1,
            (row < N) & (cs >= 0),
            sem="relaxed",
        )


def write_compact_cache(
    latent, index, latent_slots, index_slots, latent_cache, index_cache, hot_tags=None
):
    if latent.shape[-1] != 256 or index.shape[-1] != 128:
        raise ValueError("Unexpected compact dimensions")
    if latent_cache.shape[2:] != (1, 256) or index_cache.shape[2:] != (1, 128):
        raise ValueError("Unexpected paged layout")
    if latent.shape[0] != index.shape[0]:
        raise ValueError("Different token counts")
    if any(x.stride(-1) != 1 for x in (latent, index, latent_cache, index_cache)):
        raise ValueError("Contiguous last axis required")
    _store[(triton.cdiv(latent.shape[0], 4),)](
        latent,
        index,
        latent_slots,
        index_slots,
        latent_cache,
        index_cache,
        hot_tags if hot_tags is not None else latent_cache,
        hot_tags is not None,
        hot_tags.numel() if hot_tags is not None else 1,
        latent.shape[0],
        latent_slots.numel(),
        index_slots.numel(),
        latent.stride(0),
        index.stride(0),
        latent_cache.shape[1],
        index_cache.shape[1],
        *latent_cache.stride()[:2],
        *index_cache.stride()[:2],
        num_warps=4,
    )
