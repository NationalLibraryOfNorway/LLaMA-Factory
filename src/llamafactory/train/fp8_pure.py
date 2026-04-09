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

Two operating modes depending on distributed strategy:

Single GPU / DDP (no weight sharding):
  - Weights stored persistently in fp8 buffers (1B/param)
  - Forward/backward use pre-quantized fp8 weights → maximum speed
  - Compress/materialize lifecycle managed by FP8StorageCallback

DeepSpeed ZeRO-3 / FSDP (weight sharding):
  - No persistent fp8 buffers (would be unpartitioned full copies = worse memory)
  - Weights stay as bf16 nn.Parameters (partitioned by ZeRO-3 / FSDP)
  - On-the-fly quantization to fp8 during forward (after ZeRO-3 gathers weight)
  - Still get ~2x matmul speedup from native fp8 tensor cores
  - ZeRO-3 handles memory efficiency; we handle compute efficiency

In both modes:
  - Forward: weight (e4m3) × input (e4m3) via scaled_mm → bf16 output
  - Backward dL/dX: grad_output (e5m2) × weight^T (e4m3) via scaled_mm
  - Backward dL/dW: input^T (e4m3) × grad_output (e5m2) via scaled_mm

All three matmul operations use hardware fp8 tensor cores for ~2x throughput.
Requires CUDA compute capability >= 8.9 (Ada Lovelace) or >= 9.0 (Hopper).

Precision notes:
  - Accumulation in scaled_mm is fp32 internally on Hopper (TMA), bf16 on Ada.
  - LayerNorm, Softmax, embedding, and lm_head remain in bf16 (precision-critical).
  - Only nn.Linear matmul operations are replaced with fp8.
  - Optimizer step still happens in fp32 (via FP8Adafactor or standard optimizer).
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
    """Custom autograd function for fp8 matmul using torch._scaled_mm.

    Forward:  output = scaled_mm(input_e4m3, weight_e4m3^T) → bf16
    Backward: grad_input = scaled_mm(grad_output_e5m2, weight_e4m3)
              grad_weight = scaled_mm(input_e4m3^T, grad_output_e5m2) → stored in weight_proxy.grad

    Uses a weight_proxy (the original nn.Parameter) for gradient routing via STE.
    All three matmuls use hardware fp8 tensor cores.
    Handles arbitrary leading batch dimensions by flattening to 2D.
    """

    @staticmethod
    def forward(ctx, input_bf16, weight_proxy, weight_fp8, weight_scale, bias):
        orig_shape = input_bf16.shape
        in_features = orig_shape[-1]

        # Flatten to 2D: (*, K) → (M, K)
        input_2d = input_bf16.reshape(-1, in_features)

        # Quantize input to e4m3
        input_fp8, input_scale = _quantize_e4m3(input_2d)

        # scaled_mm: (M, K) @ (K, N) → (M, N)
        # B must be column-major for cuBLASLt: use .t() without .contiguous()
        output_2d = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )

        # Reshape back: (M, N) → (*, N)
        out_features = output_2d.shape[-1]
        output = output_2d.reshape(*orig_shape[:-1], out_features)

        # Save fp8 tensors for backward (1 byte each, not bf16)
        ctx.save_for_backward(input_fp8, input_scale, weight_fp8, weight_scale)
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape

        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_fp8, input_scale, weight_fp8, weight_scale = ctx.saved_tensors

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_fp8, grad_scale = _quantize_e5m2(grad_2d)

        # grad_input = grad_output @ weight  (M,N) @ (N,K) = (M,K)
        # B must be column-major: .t().contiguous().t() makes (N,K) col-major
        grad_input_2d = torch._scaled_mm(
            grad_fp8,
            weight_fp8.t().contiguous().t(),
            scale_a=grad_scale,
            scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
        grad_input = grad_input_2d.reshape(ctx.orig_shape)

        # grad_weight = grad_output^T @ input  (N,M) @ (M,K) = (N,K)
        # B must be column-major: .t().contiguous().t() makes (M,K) col-major
        grad_weight = torch._scaled_mm(
            grad_fp8.t().contiguous(),
            input_fp8.t().contiguous().t(),
            scale_a=grad_scale,
            scale_b=input_scale,
            out_dtype=torch.bfloat16,
        )

        grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1))) if ctx.has_bias else None

        # Returns: grad_input, grad_weight_proxy (STE), grad_weight_fp8=None, grad_scale=None, grad_bias
        return grad_input, grad_weight, None, None, grad_bias


class FP8PureLinear(nn.Module):
    """Linear layer using native fp8 matmul for maximum throughput.

    Two modes:
    - Buffered (single GPU / DDP): fp8 weight stored in persistent buffer,
      bf16 weight freed between steps. Maximum memory savings + speed.
    - On-the-fly (ZeRO-3 / FSDP): no buffers, weight stays as bf16 Parameter
      managed by the distributed backend. Quantized to fp8 each forward.
      Still get ~2x matmul speed; memory handled by ZeRO-3 sharding.

    Requires CUDA compute capability >= 8.9 (Ada Lovelace / RTX 4090+).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device: Optional[torch.device] = None, use_buffers: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._use_buffers = use_buffers

        # bf16 parameter — optimizer updates this
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=torch.bfloat16))
        else:
            self.bias = None

        if use_buffers:
            # Persistent fp8 storage (single GPU / DDP only)
            self.register_buffer('_weight_fp8',
                                 torch.zeros(out_features, in_features, device=device, dtype=torch.float8_e4m3fn))
            self.register_buffer('_weight_scale',
                                 torch.ones(1, device=device, dtype=torch.float32))
            self._compressed = False
        else:
            # No buffers — ZeRO-3/FSDP mode, quantize on-the-fly
            self._weight_fp8 = None
            self._weight_scale = None
            self._compressed = False

    @classmethod
    def from_linear(cls, linear: nn.Linear, use_buffers: bool = True) -> "FP8PureLinear":
        """Create FP8PureLinear from an existing nn.Linear."""
        has_bias = linear.bias is not None
        fp8_mod = cls(linear.in_features, linear.out_features, bias=has_bias,
                      device=linear.weight.device, use_buffers=use_buffers)
        fp8_mod.weight = linear.weight
        if has_bias:
            fp8_mod.bias = linear.bias

        if use_buffers:
            # Tag for fused optimizer
            fp8_mod.weight._fp8_ref = (fp8_mod, '_weight_fp8', '_weight_scale')
            fp8_mod.compress()

        return fp8_mod

    def compress(self):
        """Quantize bf16 weight to fp8, free bf16. Only in buffered mode."""
        if not self._use_buffers or self._compressed:
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
        """Decompress fp8 to bf16. Only in buffered mode."""
        if not self._use_buffers or not self._compressed:
            return
        self.weight.data = self._weight_fp8.to(torch.bfloat16) * self._weight_scale.to(torch.bfloat16)
        self._compressed = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward using native fp8 matmul.

        In buffered mode: uses pre-compressed fp8 weight, routes gradients
        through self.weight via STE (straight-through estimator).
        In on-the-fly mode: quantizes the (gathered) bf16 weight each forward.
        """
        if self._use_buffers:
            if not self._compressed:
                self.compress()
            # Pass self.weight as proxy for gradient routing (STE)
            output = _FP8MatmulFunction.apply(
                input, self.weight, self._weight_fp8, self._weight_scale, self.bias
            )
            return output
        else:
            # On-the-fly: self.weight is bf16 (gathered by ZeRO-3 during forward)
            weight_fp8, weight_scale = _quantize_e4m3(self.weight)
            output = _FP8MatmulFunction.apply(
                input, self.weight, weight_fp8, weight_scale, self.bias
            )
            return output

    def extra_repr(self) -> str:
        mode = "buffered" if self._use_buffers else "on-the-fly"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode=pure_fp8({mode})"
        )


# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------

_SKIP_MODULES = {"lm_head", "embed_tokens", "embed_positions", "wte", "wpe",
                 "embed_vision", "embedding_projection"}


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


def convert_model_to_fp8_pure(model: nn.Module, skip_vision_tower: bool = True,
                               min_numel: int = 1024, require_alignment: int = 16,
                               force_buffers: Optional[bool] = None) -> nn.Module:
    """Replace nn.Linear modules with FP8PureLinear for native fp8 matmul.

    Args:
        model: Model to convert.
        skip_vision_tower: Skip vision tower modules.
        min_numel: Minimum parameter count for conversion.
        require_alignment: Required dimension alignment.
        force_buffers: Override buffer mode. None = auto-detect (no buffers for ZeRO-3/FSDP).

    Falls back to storage mode on GPUs without native fp8 matmul support.
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

    # Auto-detect: skip buffers for ZeRO-3/FSDP (they shard Parameters, not buffers)
    if force_buffers is not None:
        use_buffers = force_buffers
    else:
        use_buffers = not _detect_zero3()

    mode_str = "buffered" if use_buffers else "on-the-fly (ZeRO-3/FSDP detected)"
    logger.info_rank0(f"FP8 pure mode: using {mode_str} weight quantization.")

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
            # Use module attributes (not weight.numel()) because ZeRO-3 partitions
            # weight tensors into 1D shards, making numel() return the shard size
            real_numel = child.in_features * child.out_features
            if real_numel < min_numel:
                skipped += 1
                continue
            if require_alignment > 0 and (child.in_features % require_alignment != 0
                                          or child.out_features % require_alignment != 0):
                skipped += 1
                continue

            fp8_mod = FP8PureLinear.from_linear(child, use_buffers=use_buffers)
            replacements.append((module, child_name, fp8_mod))
            converted += 1

    for parent, child_name, fp8_mod in replacements:
        setattr(parent, child_name, fp8_mod)

    logger.info_rank0(
        f"FP8 pure mode: converted {converted} linear layers, skipped {skipped}. "
        f"Mode: {mode_str}."
    )
    return model


def fp8_pure_compress_all(model: nn.Module) -> None:
    """Compress all FP8PureLinear weights. Only affects buffered mode."""
    for module in model.modules():
        if isinstance(module, FP8PureLinear):
            module.compress()


def fp8_pure_materialize_all(model: nn.Module) -> None:
    """Materialize all FP8PureLinear weights to bf16. Only affects buffered mode."""
    for module in model.modules():
        if isinstance(module, FP8PureLinear):
            module.materialize()
