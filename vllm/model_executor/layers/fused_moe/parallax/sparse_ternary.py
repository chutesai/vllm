# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def validate_pair_sparse(codes):
    if codes.ndim < 2 or codes.shape[-1] % 8:
        raise ValueError(
            "aligned eight-element groups along the contraction axis required"
        )
    if not bool(((codes == -1) | (codes == 0) | (codes == 1)).all()):
        raise ValueError("weights must be ternary codes")
    active = codes.reshape(*codes.shape[:-1], -1, 4, 2).ne(0).any(-1)
    if bool((active.sum(-1) > 2).any()):
        raise ValueError("not paired 4:8 sparse: compression would change the model")


def random_pair_sparse(n, k, *, device="cuda", seed=17):
    g = torch.Generator(device=device).manual_seed(seed)
    rank = torch.rand(n, k // 8, 4, generator=g, device=device).argsort(-1)
    keep = torch.zeros_like(rank, dtype=torch.bool).scatter_(-1, rank[..., :2], True)
    signs = (
        torch.randint(
            0, 2, (n, k // 8, 4, 2), generator=g, device=device, dtype=torch.int8
        )
        * 2
        - 1
    )
    return (signs * keep[..., None]).reshape(n, k).contiguous()
