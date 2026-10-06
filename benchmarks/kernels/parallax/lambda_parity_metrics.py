# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Teacher parity uses torch.argmax's first-index policy for exact ties."""


def teacher_top1(logprobs):
    highest = max(logprobs.values())
    return min(int(token) for token, value in logprobs.items() if value == highest)
