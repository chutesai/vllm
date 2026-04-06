# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Sequence

import torch

from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import BeamSearchParams, SamplingParams

logger = init_logger(__name__)


class RequestLogger:
    def __init__(self, *, max_log_len: int | None) -> None:
        self.max_log_len = max_log_len

        if not logger.isEnabledFor(logging.INFO):
            logger.warning_once(
                "`--enable-log-requests` is set but "
                "the minimum log level is higher than INFO. "
                "No request information will be logged."
            )
        elif not logger.isEnabledFor(logging.DEBUG):
            logger.info_once(
                "`--enable-log-requests` is set but "
                "the minimum log level is higher than DEBUG. "
                "Only limited information will be logged to minimize overhead. "
                "To view more details, set `VLLM_LOGGING_LEVEL=DEBUG`."
            )

    def log_inputs(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Request %s details: prompt_len: %s, "
                "prompt_token_ids_len: %s, "
                "prompt_embeds shape: %s.",
                request_id,
                len(prompt) if prompt is not None else None,
                len(prompt_token_ids) if prompt_token_ids is not None else None,
                prompt_embeds.shape if prompt_embeds is not None else None,
            )

        logger.info(
            "Received request %s: params: %s, lora_request: %s.",
            request_id,
            params,
            lora_request,
        )

    def log_outputs(
        self,
        request_id: str,
        outputs: str,
        output_token_ids: Sequence[int] | None,
        finish_reason: str | None = None,
        is_streaming: bool = False,
        delta: bool = False,
    ) -> None:
        stream_info = ""
        if is_streaming:
            stream_info = " (streaming delta)" if delta else " (streaming complete)"

        logger.info(
            "Generated response %s%s: output_len: %s, "
            "output_token_count: %s, finish_reason: %s",
            request_id,
            stream_info,
            len(outputs) if outputs is not None else None,
            len(output_token_ids) if output_token_ids is not None else None,
            finish_reason,
        )
