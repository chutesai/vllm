# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned-path eager layer tracing for mesh/fork first-divergence diagnosis."""

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class LambdaTraceWorker:
    model_runner: "GPUModelRunner"
    _lambda_trace: dict[str, Any]
    _lambda_self: dict[str, dict[str, list[dict[str, torch.Tensor]]]]

    def lambda_install_trace(self, path, inputs_path=None):
        model = self.model_runner.model
        reference = torch.load(inputs_path, weights_only=True) if inputs_path else None
        self._lambda_trace = {"layers": {}}
        self._lambda_trace_handles = []
        self._lambda_trace_path = path
        for i, layer in enumerate(model.layers):
            if reference is not None:

                def before(module, inputs, index=i):
                    if inputs[0].shape[0] == 128:
                        x = reference["layers"][str(index)]["input"].squeeze(0)
                        return (x.to(inputs[0].device), *inputs[1:])

                self._lambda_trace_handles.append(
                    layer.register_forward_pre_hook(before)
                )

            def hook(module, inputs, output, index=i):
                if (
                    inputs[0].shape[0] == 128
                    and str(index) not in self._lambda_trace["layers"]
                ):
                    self._lambda_trace["layers"][str(index)] = {
                        "input": inputs[0].detach().cpu(),
                        "branch": output.detach().cpu(),
                        "output": (inputs[0] + output).detach().cpu(),
                    }

            self._lambda_trace_handles.append(layer.register_forward_hook(hook))
        original = model.compute_logits

        def logits(hidden):
            output = original(hidden)
            if "logits" not in self._lambda_trace and hidden.shape[0] >= 127:
                self._lambda_trace["logits"] = output.detach().cpu()
            return output

        model.compute_logits = logits

    def lambda_save_trace(self):
        torch.save(self._lambda_trace, self._lambda_trace_path)
        for handle in self._lambda_trace_handles:
            handle.remove()
        return self._lambda_trace_path

    def lambda_install_self_trace(self, path):
        self._lambda_self_path = path
        self._lambda_self_phase = "decode"
        self._lambda_self = {"decode": {}, "teacher": {}}
        self._lambda_self_handles = []
        model = self.model_runner.model
        modules = [(f"layer.{i}", layer) for i, layer in enumerate(model.layers)]
        modules += [
            (f"layer.0.{name}", getattr(model.layers[0], name))
            for name in ("norm", "q_proj", "k_proj", "v_proj", "o_proj")
        ]
        modules += [
            (f"layer.1.{name}", module)
            for name, module in model.layers[1].named_modules()
            if name and isinstance(module, (torch.nn.Linear,))
        ]
        modules.append(("layer.1.norm", model.layers[1].norm))
        for name, module in modules:

            def hook(module, inputs, output, label=name):
                phase = self._lambda_self_phase
                if phase == "decode" and inputs[0].shape[0] != 1:
                    return
                row = {
                    "input": inputs[0].detach().cpu(),
                    "output": output.detach().cpu(),
                }
                self._lambda_self[phase].setdefault(label, []).append(row)

            self._lambda_self_handles.append(module.register_forward_hook(hook))

    def lambda_mark_self_teacher(self):
        self._lambda_self_phase = "teacher"

    def lambda_save_self_trace(self):
        torch.save(self._lambda_self, self._lambda_self_path)
        for handle in self._lambda_self_handles:
            handle.remove()
        return self._lambda_self_path
