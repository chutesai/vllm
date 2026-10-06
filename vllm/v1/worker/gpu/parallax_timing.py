# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in CUDA diagnostics; event overhead precludes headline timing."""

import functools
from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.worker.gpu.sample.sampler import Sampler

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _events(model, label, rows):
    key = (label, rows)
    if key not in model._component_events:
        model._component_events[key] = (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
    return model._component_events[key]


def _hook_module(model, module, label):
    def before(_module, args, kwargs):
        x = args[0] if args else kwargs.get("input_ids")
        if x is None:
            x = kwargs.get("inputs_embeds")
        if x is not None:
            rows = x.shape[0]
            _module._component_last_rows = rows
            _events(model, label, rows)[0].record()

    def after(_module, _args, _kwargs, _output):
        rows = getattr(_module, "_component_last_rows", None)
        if rows is not None:
            _events(model, label, rows)[1].record()

    module.register_forward_pre_hook(before, with_kwargs=True)
    module.register_forward_hook(after, with_kwargs=True)


def install_model_timing(model):
    model._component_events = {}
    _hook_module(model, model, "model")
    for i, layer in enumerate(model.layers):
        _hook_module(model, layer, f"layer.{i}.{type(layer).__name__}")
        if type(layer).__name__ in ("EDALayer", "GDN2Layer"):
            for name, child in layer.named_modules():
                if name and (
                    isinstance(child, torch.nn.Linear)
                    or type(child).__name__ == "BatchedEDAProjection"
                ):
                    _hook_module(model, child, f"mixer.{i}.{name}")
    original = model.compute_logits

    @functools.wraps(original)
    def logits(hidden_states):
        events = _events(model, "logits", hidden_states.shape[0])
        events[0].record()
        output = original(hidden_states)
        events[1].record()
        return output

    model.compute_logits = logits


class ComponentTimingWorker:
    model_runner: "GPUModelRunner"

    def parallax_install_model_timing(self):
        install_model_timing(self.model_runner.get_model())
        return True

    def parallax_start_cpu_profile(self):
        import cProfile

        self._parallax_cpu_profile = cProfile.Profile()
        self._parallax_cpu_profile.enable()
        return True

    def parallax_stop_cpu_profile(self, path):
        import pstats
        from pathlib import Path

        self._parallax_cpu_profile.disable()
        self._parallax_cpu_profile.dump_stats(path + ".pstats")
        with Path(path).open("w") as out:
            stats = pstats.Stats(self._parallax_cpu_profile, stream=out)
            stats.sort_stats("cumulative").print_stats(60)
            stats.sort_stats("tottime").print_stats(60)
        return path

    def parallax_install_sampler_timing(self):
        model = self.model_runner.get_model()
        original = self.model_runner.sampler
        assert original is not None
        if isinstance(original, torch.nn.Module):
            _hook_module(model, original, "sampler")
        else:
            # The V2 runner uses a callable sampler rather than nn.Module.
            original_sampler: Sampler = original

            class TimedSampler:
                def __getattr__(self, name):
                    return getattr(original_sampler, name)

                def __call__(self, logits, *args, **kwargs):
                    events = _events(model, "sampler", logits.shape[0])
                    events[0].record()
                    output = original_sampler(logits, *args, **kwargs)
                    events[1].record()
                    return output

            self.model_runner.sampler = cast(Sampler, TimedSampler())
        return True

    def parallax_component_timing(self, rows):
        model = self.model_runner.get_model()
        torch.accelerator.synchronize()
        timings = {
            label: start.elapsed_time(end)
            for (label, n), (start, end) in model._component_events.items()
            if n == rows
        }
        states = {
            str(i): [str(t.dtype) for t in layer.kv_cache]
            for i, layer in enumerate(model.layers)
            if type(layer).__name__ in ("GDN2Layer", "EDALayer")
        }
        return {
            "last_step_gpu_ms": timings,
            "gdn_state_dtypes": states,
            "scope": (
                "External CUDA events, last matching shape; "
                "diagnostic overhead included"
            ),
        }
