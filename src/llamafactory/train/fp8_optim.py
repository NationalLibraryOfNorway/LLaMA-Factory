# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FP8-fused Adafactor optimizer.

Reads fp8 weights and gradients directly, performs the Adafactor update in
fp32, and writes fp8 weights back — all per-parameter. This avoids
materializing ALL weights/gradients in bf16 simultaneously.

Memory during optimizer step (31B model example):
  Standard:  62 GB (all weights in bf16) + 62 GB (all grads in bf16) = 124 GB
  FP8 fused: 31 GB (weights in fp8) + 31 GB (grads in fp8) + ~4 GB (one param in fp32) = ~66 GB

The key insight: only ONE parameter is in fp32 at any time. After each parameter
is updated, its fp32 copy is freed before the next parameter is processed.

This optimizer subsumes the FP8StorageCallback's gradient materialization and
weight compression steps — those are no longer needed when using FP8Adafactor.
"""

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .fp8_linear import (
    FP8StorageExperts,
    FP8StorageLinear,
    dequantize_from_fp8,
    quantize_to_fp8,
)
from ..extras import logging


if TYPE_CHECKING:
    from ..hparams import TrainingArguments


logger = logging.get_logger(__name__)


class FP8Adafactor(Optimizer):
    """Adafactor optimizer with native FP8 weight/gradient support.

    Drop-in replacement for HF Adafactor when used with FP8 storage mode.
    Parameters tagged with _fp8_ref (by convert_model_to_fp8_storage) are
    read/written as fp8 directly. Untagged parameters use standard bf16/fp32.

    The math is identical to HF Adafactor — same factored second moments,
    same RMS scaling, same update clipping. Only the I/O changes.
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Static helpers (identical to HF Adafactor)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    # ------------------------------------------------------------------
    # FP8 I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_weight_fp32(p):
        """Read weight into fp32, freeing bf16 if materialized.

        Returns fp32 weight tensor. For fp8 params, reads from fp8 buffers.
        For materialized bf16 params, upcasts and frees bf16 immediately.
        """
        if hasattr(p, '_fp8_ref'):
            owner, fp8_name, scale_name = p._fp8_ref
            if p.data.numel() > 0:
                # Weight is materialized (from forward/backward) — use it directly
                w_fp32 = p.data.float()
                # Free bf16 immediately to save memory
                p.data = torch.empty(0, device=p.device, dtype=p.dtype)
            else:
                # Weight is compressed — read from fp8 buffers
                fp8_w = getattr(owner, fp8_name)
                w_scale = getattr(owner, scale_name)
                w_fp32 = dequantize_from_fp8(fp8_w, w_scale, torch.float32)
            return w_fp32, True  # is_fp8=True
        else:
            if p.dtype in {torch.float16, torch.bfloat16}:
                return p.data.float(), False
            return p.data, False

    @staticmethod
    def _read_grad_fp32(p):
        """Read gradient into fp32. Handles both fp8-compressed and regular grads."""
        if hasattr(p, '_grad_fp8') and p._grad_fp8 is not None:
            grad = dequantize_from_fp8(p._grad_fp8, p._grad_scale, torch.float32)
            # Free fp8 gradient
            p._grad_fp8 = None
            p._grad_scale = None
            return grad
        elif p.grad is not None:
            grad = p.grad
            if grad.dtype in {torch.float16, torch.bfloat16}:
                grad = grad.float()
            return grad
        return None

    @staticmethod
    def _write_weight(p, w_fp32, is_fp8):
        """Write updated weight back. FP8 params write to fp8 buffers."""
        if is_fp8:
            owner, fp8_name, scale_name = p._fp8_ref
            new_fp8, new_scale = quantize_to_fp8(w_fp32, dtype=torch.float8_e4m3fn)
            getattr(owner, fp8_name).copy_(new_fp8)
            getattr(owner, scale_name).copy_(new_scale)
            # Ensure param.data stays empty (compressed)
            if p.data.numel() > 0:
                p.data = torch.empty(0, device=p.device, dtype=p.dtype)
            # Mark module as compressed
            if isinstance(owner, FP8StorageLinear):
                owner._compressed = True
            elif isinstance(owner, FP8StorageExperts):
                owner._compressed = True
        else:
            if p.dtype in {torch.float16, torch.bfloat16}:
                p.data.copy_(w_fp32.to(p.dtype))
            else:
                p.data = w_fp32

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        For FP8-managed parameters:
        1. Read fp8 weight → fp32 (only this one param in fp32 at a time)
        2. Read fp8 gradient → fp32
        3. Standard Adafactor update in fp32
        4. Quantize result back to fp8
        5. Free fp32 temporaries before next parameter
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                # Read gradient
                grad = self._read_grad_fp32(p)
                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                # Read weight (fp32, one param at a time)
                p_data_fp32, is_fp8 = self._read_weight_fp32(p)

                state = self.state[p]
                grad_shape = grad.shape
                factored, use_first_moment = self._get_options(group, grad_shape)

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad ** 2) + group["eps"][0]

                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

                p_data_fp32.add_(-update)

                # Write back (fp8 or regular)
                self._write_weight(p, p_data_fp32, is_fp8)

                # Free fp32 temporaries immediately — this is the key memory win.
                # Only one parameter is in fp32 at a time across the entire step.
                del p_data_fp32, grad, update

        return loss


def create_fp8_adafactor(model: nn.Module, training_args: "TrainingArguments") -> FP8Adafactor:
    """Create an FP8Adafactor optimizer configured from HF TrainingArguments.

    Uses the same defaults as HF Trainer would for standard Adafactor,
    with weight decay applied to non-embedding, non-bias parameters.
    """
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "layernorm" in name or "layer_norm" in name or "embed" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    lr = training_args.learning_rate
    wd = training_args.weight_decay

    # HF Trainer default for adafactor: relative_step=True, scale_parameter=True
    # Unless explicit lr is set, then relative_step=False
    relative_step = lr is None or lr == 0
    scale_parameter = relative_step

    param_groups = [
        {"params": decay_params, "weight_decay": wd},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = FP8Adafactor(
        param_groups,
        lr=None if relative_step else lr,
        relative_step=relative_step,
        scale_parameter=scale_parameter,
        warmup_init=False,
    )

    n_fp8 = sum(1 for p in model.parameters() if hasattr(p, '_fp8_ref') and p.requires_grad)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info_rank0(
        f"FP8Adafactor: {n_fp8}/{n_total} params use fused fp8 I/O. "
        f"lr={'relative' if relative_step else lr}, weight_decay={wd}"
    )

    return optimizer
