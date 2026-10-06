# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The parity scorer must compare the same tie policy as reference argmax.

This CPU unit check catches a metric error without loading either model.
"""

import pytest
import torch

from benchmarks.kernels.parallax.lambda_parity_metrics import teacher_top1


@pytest.mark.parametrize("order", [(7, 2, 9), (9, 7, 2), (2, 9, 7)])
def test_teacher_top1_matches_argmax_independent_of_dictionary_order(order):
    logits = torch.full((10,), -10.0)
    logits[2] = logits[7] = 1.0
    lp = logits.log_softmax(-1)
    returned = {token: float(lp[token]) for token in order}
    assert teacher_top1(returned) == int(logits.argmax())
