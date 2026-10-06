# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check paged Lambda attention against FP32 causal/window attention.

The contract is packed queries and paged KV -> BF16 outputs. A cached prefix
crossing 2048 catches window boundaries, page mapping and graph padding cheaply.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.ops.parallax.lambda_dense_attention import (
    dense_attention,
)


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("window", [None, 2048])
def test_lambda_paged_attention_continues_prefix_and_zeros_padding(dim, window):
    torch.manual_seed(19)
    lengths, counts = [2053, 259], [5, 3]
    page, heads, kvheads = 16, 4, 2
    pages = 129
    table = torch.randperm(2 * pages, device="cuda").reshape(2, pages).int()
    k, v = [
        torch.randn(2 * pages, page, kvheads, dim, device="cuda").bfloat16()
        for _ in range(2)
    ]
    q = torch.randn(11, heads, dim, device="cuda").bfloat16()
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 5, 8], device="cuda", dtype=torch.int32),
        seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        block_table=table,
        max_query_len=5,
    )
    actual = dense_attention(q, k, v, metadata, window, 4096)
    expected = torch.zeros_like(q)
    offset = 0
    for seq, (length, count) in enumerate(zip(lengths, counts, strict=True)):
        loc = torch.arange(length, device="cuda")
        addresses = table[seq, loc // page].long()
        key, value = [
            x[addresses, loc % page].float().repeat_interleave(2, dim=1) for x in (k, v)
        ]
        pos = length - count + torch.arange(count, device="cuda")
        mask = loc[None, :] <= pos[:, None]
        if window:
            mask &= loc[None, :] > pos[:, None] - window
        scores = torch.einsum("qhd,khd->hqk", q[offset : offset + count].float(), key)
        probs = (scores * dim**-0.5).masked_fill(~mask, -torch.inf).softmax(-1)
        expected[offset : offset + count] = torch.einsum("hqk,khd->qhd", probs, value)
        offset += count
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    assert torch.count_nonzero(actual[8:]) == 0
