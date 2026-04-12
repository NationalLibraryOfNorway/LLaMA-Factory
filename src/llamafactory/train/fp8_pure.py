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
Native FP8 matmul kernel for Ada/Hopper GPUs.

Provides _FP8MatmulFunction, a custom autograd function that performs
forward and backward matmul using scaled_mm with fp8 tensor cores.

Used by FP8StorageLinear (in fp8_linear.py) when use_native_fp8=True.
Not a standalone module -- always paired with fp8 storage lifecycle.

Matmul operations:
  Forward:  output    = scaled_mm(input_e4m3, weight_e4m3^T) -> bf16
  Backward: grad_input  = scaled_mm(grad_e5m2, weight_e4m3)
            grad_weight = scaled_mm(input_e4m3^T, grad_e5m2)

All three use hardware fp8 tensor cores for ~2x throughput.
Requires CUDA compute capability >= 8.9 (Ada Lovelace / Hopper).

Also provides utility functions:
  _check_native_fp8_support() -- GPU capability check
  _detect_zero3() -- DeepSpeed ZeRO-3 / FSDP detection
"""

from typing import Optional

import torch
from torch.autograd import Function

from ..extras import logging


logger = logging.get_logger(__name__)

_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
_E5M2_MAX = torch.finfo(torch.float8_e5m2).max     # 57344.0

# Minimum compute capability for native fp8 matmul
_MIN_CC_NATIVE_FP8 = (8, 9)  # Ada Lovelace

# Use public F.scaled_mm (PyTorch 2.7+) with fallback to private torch._scaled_mm
_scaled_mm = getattr(torch.nn.functional, "scaled_mm", None) or torch._scaled_mm


def _check_native_fp8_support() -> bool:
    """Check if current GPU supports native fp8 matmul."""
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability()
    return cc >= _MIN_CC_NATIVE_FP8


def _quantize_e4m3(tensor: torch.Tensor, scale: Optional[torch.Tensor] = None):
    """Quantize to float8_e4m3fn with dynamic per-tensor scaling."""
    amax = tensor.detach().abs().amax()
    if scale is None:
        scale = (amax.float() / _E4M3_MAX).clamp(min=1e-12)
    fp8 = (tensor / scale).to(torch.float8_e4m3fn)
    return fp8, scale


def _quantize_e5m2(tensor: torch.Tensor, scale: Optional[torch.Tensor] = None):
    """Quantize to float8_e5m2 with dynamic per-tensor scaling."""
    amax = tensor.detach().abs().amax()
    if scale is None:
        scale = (amax.float() / _E5M2_MAX).clamp(min=1e-12)
    fp8 = (tensor / scale).to(torch.float8_e5m2)
    return fp8, scale


class _FP8MatmulFunction(Function):
    """Custom autograd function for fp8 matmul using scaled_mm.

    Forward:  output = scaled_mm(input_e4m3, weight_e4m3^T) → bf16
    Backward: grad_input = scaled_mm(grad_output_e5m2, weight_e4m3)
              grad_weight = scaled_mm(input_e4m3^T, grad_output_e5m2) → stored in weight_proxy.grad

    Uses a weight_proxy (the original nn.Parameter) for gradient routing via STE.
    All three matmuls use hardware fp8 tensor cores.
    Handles arbitrary leading batch dimensions by flattening to 2D.
    """

    @staticmethod
    def forward(ctx, input_bf16, weight_proxy, weight_fp8, weight_scale, bias, module=None):
        orig_shape = input_bf16.shape
        in_features = orig_shape[-1]

        # Flatten to 2D: (*, K) → (M, K)
        input_2d = input_bf16.reshape(-1, in_features)

        # Quantize input to e4m3
        input_fp8, input_scale = _quantize_e4m3(input_2d)

        # Pre-compute column-major weight for backward (reused for both grad matmuls)
        weight_col_major = weight_fp8.t().contiguous()

        # scaled_mm: (M, K) @ (K, N) → (M, N)
        # use_fast_accum trades ~0.1% accuracy for higher throughput on Hopper
        output_2d = _scaled_mm(
            input_fp8,
            weight_col_major.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        # Reshape back: (M, N) → (*, N)
        out_features = output_2d.shape[-1]
        output = output_2d.reshape(*orig_shape[:-1], out_features)

        # Save fp8 tensors for backward (1 byte each, not bf16)
        # Cache column-major weight to avoid recomputing in backward
        ctx.save_for_backward(input_fp8, input_scale, weight_fp8, weight_scale, weight_col_major)
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape
        ctx.module = module

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_fp8, input_scale, weight_fp8, weight_scale, weight_col_major = ctx.saved_tensors

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_fp8, grad_scale = _quantize_e5m2(grad_2d)

        # grad_input = grad_output @ weight  (M,N) @ (N,K) = (M,K)
        # Reuse column-major weight cached from forward (saves .t().contiguous())
        grad_input_2d = _scaled_mm(
            grad_fp8,
            weight_col_major.t(),
            scale_a=grad_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
        grad_input = grad_input_2d.reshape(ctx.orig_shape)

        # grad_weight = grad_output^T @ input  (N,M) @ (M,K) = (N,K)
        # M (batch tokens) may not be 16-aligned; pad if needed for _scaled_mm
        M = grad_fp8.shape[0]
        pad_m = (16 - M % 16) % 16
        if pad_m > 0:
            grad_fp8_padded = torch.nn.functional.pad(grad_fp8, (0, 0, 0, pad_m))
            input_fp8_padded = torch.nn.functional.pad(input_fp8, (0, 0, 0, pad_m))
        else:
            grad_fp8_padded = grad_fp8
            input_fp8_padded = input_fp8

        # Pre-compute column-major input for _scaled_mm B argument
        input_col_major = input_fp8_padded.t().contiguous()
        grad_weight = _scaled_mm(
            grad_fp8_padded.t().contiguous(),
            input_col_major.t(),
            scale_a=grad_scale,
            scale_b=input_scale,
            out_dtype=torch.bfloat16,
        )

        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None

        module = ctx.module
        if module is not None and getattr(module.weight, "requires_grad", False):
            param = module.weight
            if getattr(param, "_fp8_grad_compression", False):
                from .fp8_linear import quantize_to_fp8, dequantize_from_fp8
                if hasattr(param, "_grad_fp8") and param._grad_fp8 is not None:
                    prev = dequantize_from_fp8(param._grad_fp8, param._grad_scale, dtype=grad_weight.dtype)
                    accumulated = prev + grad_weight
                    fp8_grad, scale = quantize_to_fp8(accumulated, dtype=torch.float8_e5m2)
                else:
                    fp8_grad, scale = quantize_to_fp8(grad_weight, dtype=torch.float8_e5m2)
                param._grad_fp8 = fp8_grad
                param._grad_scale = scale
                param.grad = None  # Free any bf16 grad if created
            else:
                # When weight is compressed (empty), materialize before setting grad
                if hasattr(module, "_compressed") and module._compressed:
                    module.materialize()
                if param.grad is not None:
                    param.grad.add_(grad_weight)
                else:
                    param.grad = grad_weight

        # Returns: grad_input, grad_weight_proxy (None), grad_weight_fp8, grad_scale, grad_bias, module
        return grad_input, None, None, None, grad_bias, None


def _detect_zero3() -> bool:
    """Detect if DeepSpeed ZeRO-3 or FSDP will be used."""
    try:
        import deepspeed
        if hasattr(deepspeed, 'zero') and hasattr(deepspeed.zero, 'Init'):
            # Check if ZeRO-3 is configured via environment
            import json
            import os
            ds_config = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE", "")
            if ds_config and os.path.exists(ds_config):
                with open(ds_config) as f:
                    config = json.load(f)
                stage = config.get("zero_optimization", {}).get("stage", 0)
                if stage == 3:
                    return True
    except ImportError:
        pass

    # Check for FSDP (also shards parameters)
    import os
    fsdp_config = os.environ.get("ACCELERATE_USE_FSDP", "false")
    if fsdp_config.lower() == "true":
        return True

    return False


