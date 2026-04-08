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
Pure FP8 training with native scaled_mm matmul (Ada/Hopper GPUs).

This module provides maximum throughput by keeping weights and activations
in fp8 and using hardware-accelerated fp8 matmul (torch._scaled_mm).
Requires CUDA compute capability >= 8.9 (Ada Lovelace) or >= 9.0 (Hopper).

Unlike storage mode (which saves memory but computes in bf16), pure mode
gets both memory AND compute benefits:
  - Forward: weight (e4m3) × input (e4m3) via scaled_mm → bf16 output
  - Backward dL/dX: grad_output (e5m2) × weight^T (e4m3) via scaled_mm
  - Backward dL/dW: input^T (e4m3) × grad_output (e5m2) via scaled_mm

Scaling strategy: per-tensor dynamic scaling with delayed scaling.
  - Each tensor has a scale factor (fp32 scalar) updated each forward pass.
  - amax history tracks the running maximum for numerically stable scaling.
  - Scale = amax / fp8_max_val, clamped to avoid zero/inf.

Precision notes:
  - Accumulation in scaled_mm is fp32 internally on Hopper (TMA), bf16 on Ada.
  - LayerNorm, Softmax, embedding, and lm_head remain in bf16 (precision-critical).
  - Only nn.Linear matmul operations are replaced with fp8.
  - Optimizer step still happens in fp32 (via FP8Adafactor or standard optimizer).

Memory layout comparison (per-parameter):
  bf16:  2B weight + 2B activation + 2B grad = 6B
  pure:  1B weight + 1B activation + 1B grad = 3B  (50% reduction + 2x matmul speed)
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Function

from ..extras import logging


logger = logging.get_logger(__name__)

_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
_E5M2_MAX = torch.finfo(torch.float8_e5m2).max     # 57344.0

# Minimum compute capability for native fp8 matmul
_MIN_CC_NATIVE_FP8 = (8, 9)  # Ada Lovelace


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
        scale = (amax / _E4M3_MAX).clamp(min=1e-12)
    fp8 = (tensor / scale).to(torch.float8_e4m3fn)
    return fp8, scale


def _quantize_e5m2(tensor: torch.Tensor, scale: Optional[torch.Tensor] = None):
    """Quantize to float8_e5m2 with dynamic per-tensor scaling."""
    amax = tensor.detach().abs().amax()
    if scale is None:
        scale = (amax / _E5M2_MAX).clamp(min=1e-12)
    fp8 = (tensor / scale).to(torch.float8_e5m2)
    return fp8, scale


class _FP8MatmulFunction(Function):
    """Custom autograd function for fp8 matmul using torch._scaled_mm.

    Forward:  output = scaled_mm(input_e4m3, weight_e4m3^T) → bf16
    Backward: grad_input = scaled_mm(grad_output_e5m2, weight_e4m3)
              grad_weight = scaled_mm(input_e4m3^T, grad_output_e5m2)

    All three matmuls use hardware fp8 tensor cores.
    Scales are per-tensor fp32 scalars, negligible memory.

    Handles arbitrary leading batch dimensions by flattening to 2D for scaled_mm
    (which only supports 2D inputs) and reshaping back.
    """

    @staticmethod
    def forward(ctx, input_bf16, weight_fp8, weight_scale, bias):
        # Save original shape for backward reshape
        orig_shape = input_bf16.shape
        in_features = orig_shape[-1]

        # Flatten to 2D: (*, K) → (M, K) where M = product of leading dims
        input_2d = input_bf16.reshape(-1, in_features)

        # Quantize input to e4m3 for forward matmul
        input_fp8, input_scale = _quantize_e4m3(input_2d)

        # Pre-transpose weight to contiguous column-major for scaled_mm
        weight_t = weight_fp8.t().contiguous()

        # scaled_mm: (M, K) @ (K, N) → (M, N)
        output_2d = torch._scaled_mm(
            input_fp8,
            weight_t,
            scale_a=input_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )

        # Reshape back: (M, N) → (*, N)
        out_features = output_2d.shape[-1]
        output = output_2d.reshape(*orig_shape[:-1], out_features)

        # Save for backward — keep fp8 tensors (1 byte each), not bf16
        ctx.save_for_backward(input_fp8, input_scale, weight_fp8, weight_scale)
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_fp8, input_scale, weight_fp8, weight_scale = ctx.saved_tensors

        # Flatten grad_output to 2D
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])

        # Quantize grad_output to e5m2 (wider range for gradients)
        grad_fp8, grad_scale = _quantize_e5m2(grad_2d)

        # grad_input = grad_output @ weight
        # (M, N) @ (N, K) = (M, K)  where weight is (out=N, in=K)
        grad_input_2d = torch._scaled_mm(
            grad_fp8,
            weight_fp8,
            scale_a=grad_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
        # Reshape back to original input shape
        grad_input = grad_input_2d.reshape(ctx.orig_shape)

        # grad_weight = grad_output^T @ input
        # (N, M) @ (M, K) = (N, K)  which is (out, in) — correct weight shape
        grad_weight = torch._scaled_mm(
            grad_fp8.t().contiguous(),
            input_fp8,
            scale_a=grad_scale,
            scale_b=input_scale,
            out_dtype=torch.bfloat16,
        )

        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None

        return grad_input, grad_weight, None, grad_bias


class FP8PureLinear(nn.Module):
    """Linear layer using native fp8 matmul for maximum throughput.

    Weights are persistently stored in float8_e4m3fn. Forward and backward
    both use torch._scaled_mm for hardware-accelerated fp8 computation.
    Only the optimizer step temporarily uses bf16/fp32.

    Requires CUDA compute capability >= 8.9 (Ada Lovelace / RTX 4090+).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Master weight in bf16 — optimizer updates this, then we requantize
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=torch.bfloat16))
        else:
            self.bias = None

        # Persistent fp8 weight (used for all matmuls)
        self.register_buffer('_weight_fp8',
                             torch.zeros(out_features, in_features, device=device, dtype=torch.float8_e4m3fn))
        self.register_buffer('_weight_scale',
                             torch.ones(1, device=device, dtype=torch.float32))

        self._compressed = False

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FP8PureLinear":
        """Create FP8PureLinear from an existing nn.Linear."""
        has_bias = linear.bias is not None
        fp8_mod = cls(linear.in_features, linear.out_features, bias=has_bias,
                      device=linear.weight.device)
        fp8_mod.weight = linear.weight
        if has_bias:
            fp8_mod.bias = linear.bias

        # Tag for fused optimizer
        fp8_mod.weight._fp8_ref = (fp8_mod, '_weight_fp8', '_weight_scale')

        # Quantize and compress
        fp8_mod.compress()
        return fp8_mod

    def compress(self):
        """Quantize bf16 weight to fp8, free bf16."""
        if self._compressed:
            return
        w = self.weight.data
        if w.numel() == 0:
            return
        fp8_data, scale = _quantize_e4m3(w)
        self._weight_fp8.copy_(fp8_data)
        self._weight_scale.copy_(scale)
        self.weight.data = torch.empty(0, device=w.device, dtype=w.dtype)
        self._compressed = True

    def materialize(self):
        """Decompress fp8 to bf16 weight data."""
        if not self._compressed:
            return
        self.weight.data = self._weight_fp8.to(torch.bfloat16) * self._weight_scale.to(torch.bfloat16)
        self._compressed = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass using native fp8 matmul.

        Unlike FP8StorageLinear which decompresses to bf16, this keeps everything
        in fp8 for the matmul. The autograd function handles backward in fp8 too.
        """
        if not self._compressed:
            self.compress()
        return _FP8MatmulFunction.apply(input, self._weight_fp8, self._weight_scale, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode=pure_fp8"
        )


# ---------------------------------------------------------------------------
# Model conversion for pure mode
# ---------------------------------------------------------------------------

# Same skip list as storage mode
_SKIP_MODULES = {"lm_head", "embed_tokens", "embed_positions", "wte", "wpe",
                 "embed_vision", "embedding_projection"}


def convert_model_to_fp8_pure(model: nn.Module, skip_vision_tower: bool = True,
                               min_numel: int = 1024, require_alignment: int = 16) -> nn.Module:
    """Replace nn.Linear modules with FP8PureLinear for native fp8 matmul.

    Requires CUDA compute capability >= 8.9 (Ada Lovelace).
    Falls back to storage mode with a warning on older GPUs.
    """
    if not _check_native_fp8_support():
        cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        logger.warning_rank0(
            f"GPU compute capability {cc} < 8.9: native fp8 matmul not supported. "
            f"Falling back to fp8 storage mode (memory savings only, no compute speedup)."
        )
        from .fp8_linear import convert_model_to_fp8_storage
        return convert_model_to_fp8_storage(model, skip_vision_tower=skip_vision_tower,
                                             min_numel=min_numel, require_alignment=require_alignment)

    converted = 0
    skipped = 0
    replacements = []

    for name, module in model.named_modules():
        if skip_vision_tower and any(vt in name for vt in ("vision_tower", "vision_model", "visual")):
            continue

        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear):
                continue

            parts = full_name.split(".")
            if any(skip in part for skip in _SKIP_MODULES for part in parts):
                skipped += 1
                continue
            if child.weight.numel() < min_numel:
                skipped += 1
                continue
            if require_alignment > 0 and (child.in_features % require_alignment != 0
                                          or child.out_features % require_alignment != 0):
                skipped += 1
                continue

            fp8_mod = FP8PureLinear.from_linear(child)
            replacements.append((module, child_name, fp8_mod))
            converted += 1

    for parent, child_name, fp8_mod in replacements:
        setattr(parent, child_name, fp8_mod)

    logger.info_rank0(
        f"FP8 pure mode: converted {converted} linear layers to native fp8 matmul, "
        f"skipped {skipped}. Requires CC >= 8.9."
    )
    return model


def fp8_pure_compress_all(model: nn.Module) -> None:
    """Compress all FP8PureLinear weights. Call after optimizer step."""
    for module in model.modules():
        if isinstance(module, FP8PureLinear):
            module.compress()


def fp8_pure_materialize_all(model: nn.Module) -> None:
    """Materialize all FP8PureLinear weights to bf16. Call before saving."""
    for module in model.modules():
        if isinstance(module, FP8PureLinear):
            module.materialize()
