# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EDA cache contract: packed inputs -> outputs and in-place K-major state.

Compare to the sealed mesh scan to catch erase/write ordering and slot ownership
errors at kernel level, without a model load. Mesh is a read-only test oracle.
"""

from typing import cast

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.eda.parallax_eda_decode import decode
from vllm.model_executor.layers.mamba.eda.parallax_eda_prefill import EDAInputs, prefill
from vllm.model_executor.layers.mamba.gdn.parallax_conv import causal_conv


def inputs(t, h=2, d=128) -> EDAInputs:
    torch.manual_seed(17)
    q, k, v, e = [
        torch.randn(t, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    ]
    q, k, e = [F.normalize(x.float(), dim=-1).bfloat16() for x in (q, k, e)]
    g = -torch.rand(t, h, d, device="cuda") * 0.08
    b, c = [torch.rand(t, h, device="cuda") for _ in range(2)]
    return q, k, v, e, g, b, c


def scan(xs, state):
    from mesh.model.eda_kernels import _scan

    return _scan(
        *[x.unsqueeze(0) for x in xs],
        state.unsqueeze(0).float(),
        None,
        xs[0].shape[-1] ** -0.5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("block_v", [16, 64])
def test_decode_matches_mesh_and_preserves_padding(dtype, reset, block_v):
    xs = inputs(4)
    state = torch.empty_strided(
        (4, 2, 128, 128), (33792, 16384, 128, 1), device="cuda", dtype=dtype
    )
    state.copy_(torch.randn_like(state) * 0.01)
    old = state.clone()
    starts = torch.tensor([0, 1, 2, 3, 3], device="cuda", dtype=torch.int32)
    slots = torch.tensor([2, -1, 0, 1], device="cuda", dtype=torch.int32)
    resets = torch.full((4,), reset, device="cuda", dtype=torch.bool)
    out = decode(*xs, state, starts, slots, resets, block_v=block_v)
    for token, slot in [(0, 2), (2, 0)]:
        initial = torch.zeros_like(old[slot]) if reset else old[slot]
        ref, final = scan([x[token : token + 1] for x in xs], initial)
        torch.testing.assert_close(out[token].float(), ref[0, 0], atol=0.001, rtol=0.01)
        torch.testing.assert_close(
            state[slot].float(), final[0].to(dtype).float(), atol=2e-5, rtol=0.005
        )
    assert torch.count_nonzero(out[1]) == 0
    assert torch.count_nonzero(out[3]) == 0
    assert torch.equal(state[1], old[1])
    assert torch.equal(state[3], old[3])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_multistep_decode_matches_flashqla_and_chunk_continuation(dtype):
    xs = inputs(160)
    state = torch.randn(1, 2, 128, 128, device="cuda", dtype=dtype) * 0.01
    a, b, c = state.clone(), state.clone(), state.clone()
    slot = torch.tensor([0], device="cuda", dtype=torch.int32)
    reset = torch.tensor([False], device="cuda")
    starts = torch.tensor([0, 160], device="cuda", dtype=torch.int32)
    expected = prefill(*xs, a, starts, slot, reset)
    outputs = []
    for t in range(160):
        outputs.append(
            decode(
                *[x[t : t + 1] for x in xs], b, starts.new_tensor([0, 1]), slot, reset
            )
        )
    actual = torch.cat(outputs)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)
    torch.testing.assert_close(a, b, atol=0.005, rtol=0.03)
    chunks = []
    for begin, end in [(0, 65), (65, 160)]:
        chunks.append(
            prefill(
                *cast(EDAInputs, [x[begin:end] for x in xs]),
                c,
                starts.new_tensor([0, end - begin]),
                slot,
                reset,
            )
        )
    torch.testing.assert_close(torch.cat(chunks), expected, atol=0.002, rtol=0.03)
    torch.testing.assert_close(a, c, atol=0.005, rtol=0.03)


def test_conv_continuation_rounds_before_silu():
    torch.manual_seed(4)
    x = torch.randn(70, 16, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(16, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.zeros(1, 16, 3, device="cuda", dtype=x.dtype)
    slot = torch.tensor([0], device="cuda", dtype=torch.int32)
    parts = []
    for begin, end in [(0, 31), (31, 70)]:
        parts.append(
            causal_conv(
                x[begin:end],
                w,
                state,
                slot.new_tensor([0, end - begin]),
                slot,
                torch.tensor([begin == 0], device="cuda"),
                parallel=True,
                round_before_silu=True,
            )
        )
    ref = F.silu(F.conv1d(F.pad(x.T[None], (3, 0)), w[:, None], groups=16)).squeeze(0).T
    torch.testing.assert_close(torch.cat(parts), ref, atol=0.002, rtol=0.01)
    assert torch.equal(state[0], x[-3:].T)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_decode_cuda_graph_uses_live_slot_and_reset_buffers(dtype):
    xs = inputs(1)
    state = torch.randn(2, 2, 128, 128, device="cuda", dtype=dtype) * 0.01
    slots = torch.tensor([0], device="cuda", dtype=torch.int32)
    starts = slots.new_tensor([0, 1])
    reset = torch.tensor([False], device="cuda")
    decode(*xs, state, starts, slots, reset)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = decode(*xs, state, starts, slots, reset)
    old = state.clone()
    slots.fill_(1)
    reset.fill_(True)
    graph.replay()
    ref, final = scan(xs, torch.zeros_like(state[1]))
    torch.testing.assert_close(out[0].float(), ref[0, 0], atol=0.001, rtol=0.01)
    torch.testing.assert_close(state[1], final[0].to(dtype), atol=2e-5, rtol=0.005)
    assert torch.equal(state[0], old[0])


def test_ragged_prefill_gathers_reset_and_scatters_final_states():
    xs = inputs(160)
    state = torch.randn(4, 2, 128, 128, device="cuda") * 0.01
    old = state.clone()
    starts = torch.tensor([0, 65, 160], device="cuda", dtype=torch.int32)
    slots = starts.new_tensor([2, 0])
    reset = torch.tensor([True, False], device="cuda")
    out = prefill(*xs, state, starts, slots, reset)
    for begin, end, slot, clear in [(0, 65, 2, True), (65, 160, 0, False)]:
        initial = torch.zeros_like(old[slot]) if clear else old[slot]
        ref, final = scan([x[begin:end] for x in xs], initial)
        torch.testing.assert_close(
            out[begin:end].float(), ref[0], atol=0.002, rtol=0.03
        )
        torch.testing.assert_close(state[slot], final[0], atol=0.005, rtol=0.03)
    assert torch.equal(state[1], old[1])
    assert torch.equal(state[3], old[3])


def test_kappa_conv_retains_unrounded_silu_input():
    torch.manual_seed(6)
    x = torch.randn(20, 16, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(16, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.zeros(1, 16, 3, device="cuda", dtype=x.dtype)
    slot = torch.tensor([0], device="cuda", dtype=torch.int32)
    actual = causal_conv(
        x, w, state, slot.new_tensor([0, 20]), slot, torch.tensor([True], device="cuda")
    )
    ref = (
        F.silu(
            F.conv1d(F.pad(x.T[None].float(), (3, 0)), w[:, None].float(), groups=16)
        )
        .bfloat16()
        .squeeze(0)
        .T
    )
    torch.testing.assert_close(actual, ref, atol=0.0005, rtol=0.01)


def test_eda_gate_and_output_norm_match_mesh_fp32_operands():
    from mesh.model.eda import _OutputNorm, eda_safe_gate

    from vllm.model_executor.layers.mamba.eda.parallax_eda_pointwise import (
        gates,
        norm_gate,
    )

    torch.manual_seed(33)
    h, d = 2, 128
    mixed = torch.randn(3, 3 * h * d, device="cuda", dtype=torch.bfloat16)
    erase = torch.randn(3, h * d, device="cuda", dtype=mixed.dtype)
    f = torch.randn_like(erase) * 100
    b = torch.randn(3, h, device="cuda", dtype=mixed.dtype)
    a = torch.tensor([-3.0, 4.0], device="cuda")
    dt = torch.randn(h * d, device="cuda", dtype=mixed.dtype)
    result = gates(mixed, erase, f, b, b, a, dt, h, d)
    expected = eda_safe_gate(f.reshape(1, 3, h, d), dt, a)
    torch.testing.assert_close(result[4], expected[0], atol=1e-6, rtol=1e-6)
    q = mixed.reshape(3, 3, h, d)[:, 0]
    assert torch.equal(result[0], F.normalize(q.float(), dim=-1).bfloat16())
    y = torch.randn(3, h, d, device="cuda", dtype=mixed.dtype)
    gate = torch.randn_like(y)
    oracle = _OutputNorm(d).cuda()
    oracle.weight.data = torch.randn(d, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        norm_gate(y, gate, oracle.weight), oracle(y, gate), atol=0, rtol=0
    )


@pytest.mark.parametrize("tokens", [1, 4, 256, 2048])
def test_fused_pointwise_preserves_preparation_and_output_rounding(tokens):
    """Normalize/gate in FP32, including zero vectors and offset input views."""
    from vllm.model_executor.layers.mamba.eda import (
        parallax_eda_fused as fused,
    )
    from vllm.model_executor.layers.mamba.eda import (
        parallax_eda_pointwise as eager,
    )

    torch.manual_seed(31)
    h, d = 9, 128
    mixed = torch.randn(tokens + 1, 3 * h * d, device="cuda", dtype=torch.bfloat16)[1:]
    mixed[0] = 0
    erase, f = [
        torch.randn(tokens, h * d, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    erase[0] = 0
    b, c = [
        torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    a = torch.randn(h, device="cuda")
    dt = torch.randn(h * d, device="cuda", dtype=torch.bfloat16)
    args = (mixed, erase, f, b, c, a, dt, h, d)
    expected, actual = eager.gates(*args), fused.gates(*args)
    for ref, out in zip(expected, actual, strict=True):
        torch.testing.assert_close(out, ref, atol=0.001, rtol=0.005)
    y, gate = [
        torch.randn(tokens, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    weight = torch.randn(d, device="cuda")
    torch.testing.assert_close(
        fused.norm_gate(y, gate, weight),
        eager.norm_gate(y, gate, weight),
        atol=0.015625,
        rtol=0.008,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("block_v", [16, 64, 128])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("grid_order", ["sequence", "head"])
def test_gated_decode_preserves_strided_inputs_live_slots_and_state(
    dtype, block_v, num_warps, grid_order
):
    """Fusing gates must retain normalization rounding and request ownership."""
    from vllm.model_executor.layers.mamba.eda.parallax_eda_fused import gates
    from vllm.model_executor.layers.mamba.eda.parallax_eda_gated import decode_gated

    torch.manual_seed(39)
    t, h, d = 4, 9, 128
    mixed = torch.randn(t + 1, 3 * h * d, device="cuda", dtype=torch.bfloat16)[1:]
    erase, f = [
        torch.randn(t, h * d, device="cuda", dtype=mixed.dtype) for _ in range(2)
    ]
    biases = torch.randn(t, 2 * h + 17, device="cuda", dtype=mixed.dtype)
    b, c = biases[:, :h], biases[:, h : 2 * h]
    a = torch.randn(h, device="cuda")
    dt = torch.randn(h * d, device="cuda", dtype=mixed.dtype)
    mixed[0] = 0
    erase[0] = 0
    f[0] *= 100
    state = torch.empty_strided(
        (4, h, d, d), (h * d * d + 1024, d * d, d, 1), device="cuda", dtype=dtype
    )
    state.copy_(torch.randn_like(state) * 0.01)
    expected_state = state.clone()
    old = state.clone()
    starts = torch.tensor([0, 1, 2, 3, 3], device="cuda", dtype=torch.int32)
    slots = starts.new_tensor([2, -1, 0, 1])
    reset = torch.tensor([True, False, False, False], device="cuda")
    args = (mixed, erase, f, b, c, a, dt)
    expected = decode(
        *gates(*args, h, d), expected_state, starts, slots, reset, block_v=block_v
    )
    actual = decode_gated(
        *args,
        state,
        starts,
        slots,
        reset,
        block_v=block_v,
        num_warps=num_warps,
        grid_order=grid_order,
    )
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=0.005)
    assert torch.equal(state[1], old[1])
    assert torch.equal(state[3], old[3])


@pytest.mark.parametrize("tokens", [1, 4, 32, 128, 256])
def test_stable_joined_gemm_preserves_bias_epilogue_and_offset_rows(tokens):
    """Bias must be added in FP32 before storing joined projections in BF16."""
    from vllm.model_executor.layers.mamba.eda.parallax_eda_linear import linear

    torch.manual_seed(67)
    width, outputs = 1152, 3890
    x = torch.randn(tokens + 1, width + 8, device="cuda", dtype=torch.bfloat16)[
        1:, :width
    ]
    weight = torch.randn(outputs, width, device="cuda", dtype=x.dtype) * 0.02
    bias = torch.zeros(outputs, device="cuda", dtype=x.dtype)
    bias[3744:3762] = torch.linspace(-0.3, 0.3, 18, device="cuda", dtype=x.dtype)
    expected = F.linear(x.float(), weight.float(), bias.float()).bfloat16()
    actual = linear(x, weight, bias)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
