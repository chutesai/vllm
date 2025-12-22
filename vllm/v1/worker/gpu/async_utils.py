# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import torch

from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_stream = copy_stream
        self.copy_event = copy_event

        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(default_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors: LogprobsTensors | None = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            else:
                self.logprobs_tensors = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            else:
                self.num_nans = None
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}
            if self.model_runner_output.prompt_logprobs_dict:
                for k, v in self.model_runner_output.prompt_logprobs_dict.items():
                    if v is not None:
                        self.prompt_logprobs_dict[k] = v.to_cpu_nonblocking()
                    else:
                        self.prompt_logprobs_dict[k] = None
            self.copy_event.record(self.copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_reqs = len(sampled_token_ids)
        num_sampled_tokens = self.num_sampled_tokens_np.tolist()
        for i in range(num_reqs):
            del sampled_token_ids[i][num_sampled_tokens[i] :]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.num_nans is not None:
            num_nans = self.num_nans.tolist()
            self.model_runner_output.num_nans_in_logits = {
                req_id: num_nans[i]
                for i, req_id in enumerate(self.model_runner_output.req_ids)
            }

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        return self.model_runner_output


@contextmanager
def async_barrier(event: torch.cuda.Event | None):
    if event is not None:
        event.synchronize()
    try:
        yield
    finally:
        if event is not None:
            event.record()


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


class AsyncOutputCC(AsyncModelRunnerOutput):
    """Async output optimized for confidential computing environments.

    In confidential computing (Intel TDX + NVIDIA protected PCIe),
    cudaMemcpyAsync becomes effectively synchronous due to memory
    encryption overhead. This class moves the entire D2H copy operation
    to a dedicated worker thread to avoid blocking the main thread.

    Unlike AsyncOutput which starts copies in __init__ (blocking in CC),
    this class defers copies to futures resolved in get_output().
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        copy_executor: ThreadPoolExecutor,
    ):
        # Retain references to GPU tensors to keep them alive until copy
        # completes in the worker thread.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.copy_executor = copy_executor

        # Record event on current stream to ensure GPU work is complete
        # before worker thread starts copying.
        self.copy_event = torch.cuda.Event()
        self.copy_event.record(torch.cuda.current_stream())

        # Submit copy tasks to worker thread. The worker will synchronize
        # on the event before copying, ensuring GPU computation is done.
        self.sampled_token_ids_future: Future[np.ndarray] = copy_executor.submit(
            self._sync_copy_to_np, sampler_output.sampled_token_ids
        )

        if sampler_output.logprobs_tensors is not None:
            self.logprobs_tensors_future: Future[LogprobsTensors] | None = (
                copy_executor.submit(
                    self._sync_copy_logprobs, sampler_output.logprobs_tensors
                )
            )
        else:
            self.logprobs_tensors_future = None

        if sampler_output.num_nans is not None:
            self.num_nans_future: Future[np.ndarray] | None = copy_executor.submit(
                self._sync_copy_to_np, sampler_output.num_nans
            )
        else:
            self.num_nans_future = None

        self.num_sampled_tokens_future: Future[np.ndarray] = copy_executor.submit(
            self._sync_copy_to_np, num_sampled_tokens
        )

        self.prompt_logprobs_futures: dict[
            str, Future[LogprobsTensors | None] | None
        ] = {}
        if self.model_runner_output.prompt_logprobs_dict:
            for k, v in self.model_runner_output.prompt_logprobs_dict.items():
                if v is not None:
                    self.prompt_logprobs_futures[k] = copy_executor.submit(
                        self._sync_copy_logprobs, v
                    )
                else:
                    self.prompt_logprobs_futures[k] = None

    def _sync_copy_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        """Copy tensor to CPU numpy array, synchronizing first.

        This runs in the worker thread. We synchronize on the event to
        ensure all GPU work is complete before starting the copy.
        """
        self.copy_event.synchronize()
        return tensor.cpu().numpy()

    def _sync_copy_logprobs(self, logprobs: LogprobsTensors) -> LogprobsTensors:
        """Copy LogprobsTensors to CPU, synchronizing first."""
        self.copy_event.synchronize()
        return LogprobsTensors(
            logprobs.logprob_token_ids.cpu(),
            logprobs.logprobs.cpu(),
            logprobs.selected_token_ranks.cpu(),
        )

    def get_output(self) -> ModelRunnerOutput:
        # Resolve futures - this blocks until worker thread completes,
        # but the worker has already done the GPU sync and copy.
        sampled_token_ids_np = self.sampled_token_ids_future.result()
        num_sampled_tokens_np = self.num_sampled_tokens_future.result()

        # Convert to list format expected by downstream code.
        sampled_token_ids: list[list[int]] = sampled_token_ids_np.tolist()
        num_reqs = len(sampled_token_ids)
        num_sampled_tokens = num_sampled_tokens_np.tolist()
        for i in range(num_reqs):
            del sampled_token_ids[i][num_sampled_tokens[i] :]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.num_nans_future is not None:
            num_nans = self.num_nans_future.result().tolist()
            self.model_runner_output.num_nans_in_logits = {
                req_id: num_nans[i]
                for i, req_id in enumerate(self.model_runner_output.req_ids)
            }

        if self.logprobs_tensors_future is not None:
            logprobs_tensors = self.logprobs_tensors_future.result()
            self.model_runner_output.logprobs = logprobs_tensors.tolists()

        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}
        for k, future in self.prompt_logprobs_futures.items():
            if future is not None:
                prompt_logprobs_dict[k] = future.result()
            else:
                prompt_logprobs_dict[k] = None
        self.model_runner_output.prompt_logprobs_dict = prompt_logprobs_dict

        return self.model_runner_output
