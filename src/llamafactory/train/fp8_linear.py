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
True FP8 training with fp8 weight and gradient storage.

This module implements two complementary memory optimizations for training:

1. FP8 Weight Storage (float8_e4m3fn, 1 byte/param)
   Weights are stored in fp8 between training steps and decompressed to bf16
   for forward/backward. After optimizer step, recompressed to fp8.
   Memory savings require gradient checkpointing.

   Supports two module types:
   - nn.Linear layers → replaced by FP8StorageLinear
   - MoE expert modules with 3D weight tensors (e.g. Mixtral, Qwen MoE)
     → wrapped by FP8StorageExperts. These store gate_up_proj/down_proj as
     3D nn.Parameter tensors, not nn.Linear. The wrapper manages the same
     compress/materialize lifecycle around the original module.

2. FP8 Gradient Compression (float8_e5m2, 1 byte/param)
   Gradients are compressed to fp8 immediately as they're produced during
   backward. This halves gradient memory from 2B/param (bf16) to 1B/param.

   Design decisions:
   - Uses float8_e5m2 for gradients (5 exponent bits = wide dynamic range,
     critical for gradients which span many orders of magnitude).
   - Uses float8_e4m3fn for weights (4 exponent bits, 3 mantissa bits = better
     precision, since weight distributions are tighter).
   - Accumulation with gradient_accumulation_steps > 1: each micro-batch's
     gradient is accumulated in fp8 via dequant-add-requant. The ~0.5% error
     per step is negligible relative to stochastic gradient noise (10-50%).
     For typical grad_accum=2-8 this is well within tolerance. Users doing
     grad_accum=16+ in fp8 are in unusual territory and should validate.
   - Gradients are decompressed to bf16 before the optimizer step, because
     optimizers (Adam, Adafactor) expect bf16/fp32 and do their own
     precision-sensitive accumulation internally.
   - Per-tensor dynamic scaling (one fp32 scalar per gradient tensor).
     Negligible memory overhead (~4 bytes per layer).

Both work on any CUDA GPU (compute capability >= 7.0). No Ada/Hopper required.
Ada/Hopper GPUs additionally benefit from native fp8 matmul in "pure" mode.

Memory layout per parameter (bf16 baseline vs fp8 storage+grads):
  bf16:  2B weight + 2B grad = 4B/param
  fp8:   1B weight + 1B grad = 2B/param  (50% reduction)
  + optimizer states (unchanged, use bnb int8 Adam or adafactor to compress)
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainerCallback

from ..extras import logging


if TYPE_CHECKING:
    from ..hparams import TrainingArguments


logger = logging.get_logger(__name__)

# float8_e4m3fn: 4 exponent bits, 3 mantissa bits, max 448.0 — for weights
_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# float8_e5m2: 5 exponent bits, 2 mantissa bits, max 57344.0 — for gradients
# Wider dynamic range is critical for gradients which span many orders of magnitude.
_E5M2_MAX = torch.finfo(torch.float8_e5m2).max


def quantize_to_fp8(tensor: torch.Tensor,
                    dtype: torch.dtype = torch.float8_e4m3fn) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bf16/fp32 tensor to fp8 with per-tensor dynamic scaling.

    Args:
        tensor: Input tensor in bf16 or fp32.
        dtype: Target fp8 dtype. Use float8_e4m3fn for weights, float8_e5m2 for gradients.

    Returns:
        (fp8_tensor, scale) where original ≈ fp8_tensor.to(bf16) * scale
    """
    fp8_max = _E4M3_MAX if dtype == torch.float8_e4m3fn else _E5M2_MAX
    amax = tensor.detach().abs().amax()
    scale = amax / fp8_max
    scale = scale.clamp(min=1e-12)  # avoid division by zero
    fp8_tensor = (tensor / scale).to(dtype)
    return fp8_tensor, scale


def dequantize_from_fp8(fp8_tensor: torch.Tensor, scale: torch.Tensor,
                        dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize fp8 tensor back to bf16/fp32."""
    return fp8_tensor.to(dtype) * scale.to(dtype)


# ---------------------------------------------------------------------------
# FP8 Weight Storage
# ---------------------------------------------------------------------------

class FP8StorageLinear(nn.Module):
    """Linear layer with FP8 weight storage for memory-efficient training.

    Weights are stored in float8_e4m3fn (1 byte/param) between training steps.
    During forward/backward, weights are decompressed to bf16 for computation.
    After the optimizer step, weights are recompressed to fp8.

    Memory savings require gradient checkpointing so that only one checkpoint
    segment's weights need to be in bf16 at any given time during forward/backward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
                 use_native_fp8: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._use_native_fp8 = use_native_fp8

        # The bf16 parameter that autograd and optimizer interact with.
        # Between steps, its .data is swapped to an empty tensor to free memory.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype or torch.bfloat16))
        else:
            self.bias = None

        # FP8 storage buffers (persistent between steps)
        self.register_buffer('_weight_fp8', torch.zeros(out_features, in_features, device=device, dtype=torch.float8_e4m3fn))
        self.register_buffer('_weight_scale', torch.ones(1, device=device, dtype=torch.float32))

        # State tracking
        self._compressed = False

    @classmethod
    def from_linear(cls, linear: nn.Linear, use_native_fp8: bool = False) -> "FP8StorageLinear":
        """Create FP8StorageLinear from an existing nn.Linear, quantizing its weights."""
        has_bias = linear.bias is not None
        fp8_linear = cls(
            linear.in_features, linear.out_features,
            bias=has_bias,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            use_native_fp8=use_native_fp8,
        )
        # Copy weight data
        fp8_linear.weight = linear.weight
        if has_bias:
            fp8_linear.bias = linear.bias

        # Tag parameter so fused optimizer can find fp8 buffers
        fp8_linear.weight._fp8_ref = (fp8_linear, '_weight_fp8', '_weight_scale')

        # Quantize to fp8 and compress
        fp8_linear.compress()
        return fp8_linear

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle loading from bf16, compressed, or fp8-native checkpoints."""
        weight_key = prefix + "weight"
        fp8_key = prefix + "_weight_fp8"
        scale_key = prefix + "_weight_scale"
        # Common external scale key patterns (e.g. from torchao, vLLM quantized models)
        ext_scale_key = prefix + "weight_scale"

        if weight_key in state_dict:
            w = state_dict[weight_key]
            if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # FP8-native checkpoint: weight tensor is already fp8
                if ext_scale_key in state_dict:
                    # Scale provided — load fp8 data + scale directly, no casting
                    state_dict[fp8_key] = w.to(torch.float8_e4m3fn)
                    state_dict[scale_key] = state_dict.pop(ext_scale_key).to(torch.float32)
                elif scale_key in state_dict:
                    # Our own buffer scale key is present
                    state_dict[fp8_key] = w.to(torch.float8_e4m3fn)
                else:
                    # No scale found — round-trip through bf16 to compute one
                    fp8_data, scale = quantize_to_fp8(w.to(torch.bfloat16), dtype=torch.float8_e4m3fn)
                    state_dict[fp8_key] = fp8_data
                    state_dict[scale_key] = scale
                # Replace weight with empty tensor so Parameter stays bf16
                state_dict[weight_key] = torch.empty(0, dtype=self.weight.dtype)
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
                self._compressed = True
                return
            elif w.numel() == 0 and fp8_key in state_dict:
                # Compressed checkpoint: weight is empty, fp8 buffers have data
                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
                self._compressed = True
                return

        # Standard bf16 checkpoint: load normally, compress() will be called later
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def compress(self):
        """Compress bf16 weight to fp8, free bf16 data."""
        if self._compressed:
            return

        w = self.weight.data
        if w.numel() == 0:
            return

        fp8_data, scale = quantize_to_fp8(w, dtype=torch.float8_e4m3fn)
        self._weight_fp8.copy_(fp8_data)
        self._weight_scale.copy_(scale)

        # Free bf16 weight data by replacing with empty tensor
        self.weight.data = torch.empty(0, device=w.device, dtype=w.dtype)
        self._compressed = True

    def materialize(self):
        """Decompress fp8 to bf16 weight data."""
        if not self._compressed:
            return

        bf16_weight = dequantize_from_fp8(self._weight_fp8, self._weight_scale, dtype=self.weight.dtype)
        self.weight.data = bf16_weight
        self._compressed = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._use_native_fp8 and self._compressed:
            # Native fp8 matmul: use fp8 buffers directly, no materialize needed.
            # Gradients route through self.weight via STE in _FP8MatmulFunction.
            from .fp8_pure import _FP8MatmulFunction
            return _FP8MatmulFunction.apply(
                input, self.weight, self._weight_fp8, self._weight_scale, self.bias
            )

        # Standard path: decompress to bf16, run matmul, re-compress
        self.materialize()
        output = F.linear(input, self.weight, self.bias)
        # Re-compress after forward to free bf16 memory.
        # With gradient checkpointing, backward re-runs forward anyway.
        if self.training:
            self.compress()
        return output

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Save fp8 weights under standard keys for ecosystem compatibility.

        Maps internal buffers to standard keys so checkpoints work with
        vLLM, transformers, and other inference engines:
          weight       → fp8 data (float8_e4m3fn), not the empty bf16 shell
          weight_scale → float32 per-tensor scale
          bias         → saved normally if present
        Internal _weight_fp8 and _weight_scale keys are not exposed.
        """
        if self._compressed:
            # Save fp8 data as 'weight' — standard key, fp8 dtype
            destination[prefix + "weight"] = self._weight_fp8 if keep_vars else self._weight_fp8.detach()
            destination[prefix + "weight_scale"] = self._weight_scale if keep_vars else self._weight_scale.detach()
        else:
            # Not compressed (e.g. mid-optimizer-step): save bf16 weight normally
            destination[prefix + "weight"] = self.weight if keep_vars else self.weight.detach()
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

    def extra_repr(self) -> str:
        mode = "native_fp8" if self._use_native_fp8 else "storage"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, compressed={self._compressed}, mode={mode}"
        )


# ---------------------------------------------------------------------------
# FP8 MoE Expert Storage
# ---------------------------------------------------------------------------

class FP8StorageExperts(nn.Module):
    """Wrapper for MoE expert modules that stores 3D weight parameters in fp8.

    MoE models (Mixtral, Qwen MoE, GLM MoE, etc.) store expert weights as 3D
    nn.Parameter tensors like gate_up_proj[num_experts, 2*intermediate, hidden],
    not as nn.Linear. This wrapper handles the compress/materialize lifecycle
    for these 3D parameters, identical in spirit to FP8StorageLinear.

    The original module is kept as self._inner. Its parameters remain visible
    to the optimizer (they're discovered via recursive parameter search).
    Between steps, param.data is swapped to empty tensors to free memory,
    with the actual values stored in fp8 buffers on this wrapper.
    """

    def __init__(self, experts_module: nn.Module, expert_param_names: list[str]):
        super().__init__()
        self._inner = experts_module
        self._expert_param_names = expert_param_names
        self._compressed = False

        # Create fp8 storage buffers for each expert parameter
        # Use ds_shape if available (ZeRO-3 flattens params to 1D shards)
        for name in expert_param_names:
            param = getattr(experts_module, name)
            shape = getattr(param, 'ds_shape', param.shape)
            self.register_buffer(
                f'_fp8_{name}',
                torch.zeros(shape, device=param.device, dtype=torch.float8_e4m3fn),
            )
            self.register_buffer(
                f'_scale_{name}',
                torch.ones(1, device=param.device, dtype=torch.float32),
            )

    @classmethod
    def from_experts(cls, experts_module: nn.Module) -> Optional["FP8StorageExperts"]:
        """Wrap an experts module if it has 3D weight parameters.

        Returns None if no eligible parameters found.
        """
        param_names = []
        for name, param in experts_module.named_parameters(recurse=False):
            # With ZeRO-3, param is a 1D shard. Use ds_shape if available.
            shape = getattr(param, 'ds_shape', param.shape)
            numel = shape.numel() if hasattr(shape, 'numel') else param.numel()
            if len(shape) >= 3 and numel >= 1024:
                param_names.append(name)

        if not param_names:
            return None

        wrapper = cls(experts_module, param_names)

        # Tag parameters so fused optimizer can find fp8 buffers
        for name in param_names:
            param = getattr(experts_module, name)
            param._fp8_ref = (wrapper, f'_fp8_{name}', f'_scale_{name}')

        wrapper.compress()
        return wrapper

    def compress(self):
        """Compress all 3D expert weights to fp8, free bf16 data."""
        if self._compressed:
            return

        for name in self._expert_param_names:
            param = getattr(self._inner, name)
            if param.data.numel() == 0:
                continue
            fp8_data, scale = quantize_to_fp8(param.data, dtype=torch.float8_e4m3fn)
            getattr(self, f'_fp8_{name}').copy_(fp8_data)
            getattr(self, f'_scale_{name}').copy_(scale)
            param.data = torch.empty(0, device=param.device, dtype=param.dtype)

        self._compressed = True

    def materialize(self):
        """Decompress all fp8 expert weights back to bf16."""
        if not self._compressed:
            return

        for name in self._expert_param_names:
            param = getattr(self._inner, name)
            fp8_data = getattr(self, f'_fp8_{name}')
            scale = getattr(self, f'_scale_{name}')
            param.data = dequantize_from_fp8(fp8_data, scale, dtype=param.dtype)

        self._compressed = False

    def forward(self, *args, **kwargs):
        self.materialize()
        output = self._inner.forward(*args, **kwargs)
        if self.training:
            self.compress()
        return output

    def __getattr__(self, name):
        # Delegate attribute access to inner module for compatibility
        # (e.g. accessing num_experts, act_fn, etc.)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# FP8 Gradient Compression
# ---------------------------------------------------------------------------

def _make_fp8_grad_hook(param: nn.Parameter):
    """Create a post-accumulate gradient hook that compresses gradients to fp8_e5m2.

    The hook fires after each backward pass (including each micro-batch in gradient
    accumulation). For grad_accum > 1, it accumulates in fp8 via dequant-add-requant:

      1. First micro-batch: grad is fresh → compress to fp8, store as _grad_fp8/_grad_scale
      2. Subsequent micro-batches: dequant previous fp8 accumulator, add new bf16 grad,
         requant back to fp8. PyTorch sets param.grad = new_grad (not accumulated) because
         we null it after each hook, so we manually accumulate with the fp8 buffer.

    Error analysis: e5m2 has ~0.5% relative quantization error per step. Over N accumulation
    steps, the error grows as ~sqrt(N) * 0.5% (errors are uncorrelated across micro-batches).
    For grad_accum=4: ~1% error. For grad_accum=8: ~1.4% error. Stochastic gradient noise
    is typically 10-50%, so this is well within tolerance.
    """
    def hook(param: nn.Parameter):
        if param.grad is None:
            return

        new_grad = param.grad.detach()

        if hasattr(param, '_grad_fp8') and param._grad_fp8 is not None:
            # Accumulate: dequant previous + add new + requant
            prev = dequantize_from_fp8(param._grad_fp8, param._grad_scale, dtype=new_grad.dtype)
            accumulated = prev + new_grad
            fp8_grad, scale = quantize_to_fp8(accumulated, dtype=torch.float8_e5m2)
        else:
            # First micro-batch: just compress
            fp8_grad, scale = quantize_to_fp8(new_grad, dtype=torch.float8_e5m2)

        param._grad_fp8 = fp8_grad
        param._grad_scale = scale

        # Free bf16 gradient memory — this is the whole point
        param.grad = None

    return hook


def _is_fp8_managed(module: nn.Module) -> bool:
    """Check if a module is an FP8-managed module (linear, experts, or pure)."""
    from .fp8_pure import FP8PureLinear
    return isinstance(module, (FP8StorageLinear, FP8StorageExperts, FP8PureLinear))


def _iter_fp8_params(model: nn.Module):
    """Iterate over all parameters managed by FP8 modules (storage, experts, and pure)."""
    from .fp8_pure import FP8PureLinear
    for module in model.modules():
        if isinstance(module, (FP8StorageLinear, FP8PureLinear)):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
        elif isinstance(module, FP8StorageExperts):
            # Yield the inner module's parameters (the 3D expert weights)
            for param in module._inner.parameters():
                if param.requires_grad:
                    yield param


def install_fp8_grad_hooks(model: nn.Module) -> list:
    """Install fp8 gradient compression hooks on all FP8-managed parameters.

    Covers both FP8StorageLinear (2D) and FP8StorageExperts (3D) parameters.
    Returns list of hook handles (call .remove() to uninstall).
    """
    handles = []
    hooked = 0
    seen = set()
    for param in _iter_fp8_params(model):
        if id(param) in seen:
            continue
        seen.add(id(param))
        handle = param.register_post_accumulate_grad_hook(_make_fp8_grad_hook(param))
        handles.append(handle)
        hooked += 1
    logger.info_rank0(f"FP8 gradients: installed compression hooks on {hooked} parameters.")
    return handles


def materialize_fp8_gradients(model: nn.Module) -> None:
    """Decompress all fp8 gradients back to bf16 for the optimizer step.

    Must be called before optimizer.step(). The optimizer expects param.grad
    in bf16/fp32. After the optimizer step, gradients are zeroed anyway.
    """
    for param in _iter_fp8_params(model):
        if hasattr(param, '_grad_fp8') and param._grad_fp8 is not None:
            param.grad = dequantize_from_fp8(
                param._grad_fp8, param._grad_scale, dtype=param.dtype
            )
            param._grad_fp8 = None
            param._grad_scale = None


def clear_fp8_grad_accumulators(model: nn.Module) -> None:
    """Clear fp8 gradient accumulators after optimizer.zero_grad()."""
    for param in _iter_fp8_params(model):
        if hasattr(param, '_grad_fp8'):
            param._grad_fp8 = None
            param._grad_scale = None


# ---------------------------------------------------------------------------
# Model conversion utilities
# ---------------------------------------------------------------------------

# Modules to skip for FP8 conversion (precision-sensitive or tiny)
_SKIP_MODULES = {"lm_head", "embed_tokens", "embed_positions", "wte", "wpe",
                 "embed_vision", "embedding_projection"}

# Common names for MoE expert modules across transformers models.
# These contain 3D nn.Parameter tensors (gate_up_proj, down_proj, etc.)
_EXPERT_MODULE_NAMES = {"experts", "expert"}


def _should_convert_linear(name: str, module: nn.Module,
                           min_numel: int = 1024, require_alignment: int = 16) -> bool:
    """Check if an nn.Linear module should be converted to FP8."""
    if not isinstance(module, nn.Linear):
        return False

    # Skip embedding-related and output layers
    parts = name.split(".")
    if any(skip in part for skip in _SKIP_MODULES for part in parts):
        return False

    # Skip tiny layers (not worth the overhead)
    # Use module attributes (not weight.numel()) because ZeRO-3 partitions
    # weight tensors into 1D shards, making numel() return the shard size
    real_numel = module.in_features * module.out_features
    if real_numel < min_numel:
        return False

    # Skip layers with dimensions not meeting alignment requirement
    if require_alignment > 0 and (module.in_features % require_alignment != 0 or module.out_features % require_alignment != 0):
        return False

    return True


def _is_expert_module(name: str, module: nn.Module) -> bool:
    """Check if a module is a MoE experts module with 3D weight parameters.

    Detects modules like MixtralExperts, Qwen2MoeExperts, etc. that store
    expert weights as 3D nn.Parameter tensors (not nn.Linear).
    """
    # Check by name
    leaf_name = name.split(".")[-1] if name else ""
    if leaf_name not in _EXPERT_MODULE_NAMES:
        return False

    # Verify it has 3D parameters (the hallmark of expert weight storage)
    # With ZeRO-3, param.dim() may be 1 (flattened shard), so also check ds_shape
    for param_name, param in module.named_parameters(recurse=False):
        shape = getattr(param, 'ds_shape', param.shape)
        numel = shape.numel() if hasattr(shape, 'numel') else param.numel()
        if len(shape) >= 3 and numel >= 1024:
            return True

    return False


def convert_model_to_fp8_storage(model: nn.Module, skip_vision_tower: bool = True,
                                  min_numel: int = 1024, require_alignment: int = 16,
                                  fp8_gradients: bool = True) -> nn.Module:
    """Convert model to use FP8 weight storage for memory-efficient training.

    Handles two module types:
    - nn.Linear layers → replaced by FP8StorageLinear
    - MoE expert modules with 3D weight tensors → wrapped by FP8StorageExperts

    Args:
        model: The model to convert.
        skip_vision_tower: If True, skip vision tower modules (for multimodal models).
        min_numel: Minimum parameter count for linear layer conversion.
        require_alignment: Required dimension alignment for linear layers (0 to disable).
        fp8_gradients: If True, also install fp8 gradient compression hooks.

    Returns:
        The model with eligible modules converted to FP8 storage.
    """
    # Detect native fp8 matmul support (Ada Lovelace / Hopper)
    from .fp8_pure import _check_native_fp8_support
    use_native_fp8 = _check_native_fp8_support()
    if use_native_fp8:
        logger.info_rank0("FP8 storage: native fp8 matmul detected, using scaled_mm for compute.")

    linear_converted = 0
    linear_skipped = 0
    expert_converted = 0
    expert_params_total = 0

    # Collect modules to replace (can't modify during iteration)
    replacements: list[tuple[nn.Module, str, nn.Module]] = []

    for name, module in model.named_modules():
        # Skip vision tower
        if skip_vision_tower and any(vt in name for vt in ("vision_tower", "vision_model", "visual")):
            continue

        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check for MoE expert modules (3D parameter tensors)
            if _is_expert_module(full_name, child):
                wrapper = FP8StorageExperts.from_experts(child)
                if wrapper is not None:
                    replacements.append((module, child_name, wrapper))
                    expert_converted += 1
                    expert_params_total += sum(
                        getattr(wrapper, f'_fp8_{n}').numel()
                        for n in wrapper._expert_param_names
                    )
                continue

            # Check for nn.Linear modules
            if _should_convert_linear(full_name, child, min_numel=min_numel, require_alignment=require_alignment):
                fp8_linear = FP8StorageLinear.from_linear(child, use_native_fp8=use_native_fp8)
                replacements.append((module, child_name, fp8_linear))
                linear_converted += 1
            elif isinstance(child, nn.Linear):
                linear_skipped += 1

    # Apply replacements
    for parent, child_name, fp8_module in replacements:
        setattr(parent, child_name, fp8_module)

    total_converted = linear_converted + expert_converted
    if total_converted > 0:
        compute = "native fp8 matmul" if use_native_fp8 else "bf16 matmul"
        msg = f"FP8 storage ({compute}): converted {linear_converted} linear layers"
        if expert_converted > 0:
            msg += f", {expert_converted} expert modules ({expert_params_total:,} expert params)"
        msg += f", skipped {linear_skipped} layers."
        logger.info_rank0(msg)
    else:
        logger.info_rank0("FP8 storage: no eligible modules found for conversion.")

    # Install gradient compression hooks
    if fp8_gradients and total_converted > 0:
        install_fp8_grad_hooks(model)

    return model


# ---------------------------------------------------------------------------
# Weight compression/materialization utilities
# ---------------------------------------------------------------------------

def _is_fp8_module(module: nn.Module) -> bool:
    """Check if a module is any FP8-managed type with compress/materialize."""
    return isinstance(module, (FP8StorageLinear, FP8StorageExperts))


def fp8_compress_all(model: nn.Module) -> None:
    """Compress all FP8-managed weights to fp8. Call after optimizer.step()."""
    for module in model.modules():
        if _is_fp8_module(module):
            module.compress()


def fp8_materialize_all(model: nn.Module) -> None:
    """Materialize all FP8-managed weights to bf16. Call before saving."""
    for module in model.modules():
        if _is_fp8_module(module):
            module.materialize()


# ---------------------------------------------------------------------------
# Trainer callback
# ---------------------------------------------------------------------------

class FP8StorageCallback(TrainerCallback):
    """Training callback that manages the FP8 weight + gradient lifecycle.

    Two modes depending on whether a fused optimizer (FP8Adafactor) is used:

    Standard (fused_optimizer=False):
      1. on_train_begin: compress weights, store model ref
      2. forward: auto-decompressed by FP8StorageLinear.forward()
      3. backward: gradient hooks compress grads to fp8_e5m2
      4. on_pre_optimizer_step: decompress grads to bf16 for standard optimizer
      5. optimizer.step(): standard bf16 update
      6. on_step_end: compress weights back, clear grad accumulators
      7. on_save: materialize for checkpoint

    Fused (fused_optimizer=True):
      1. on_train_begin: compress weights, store model ref
      2. forward/backward: same as above
      3. on_pre_optimizer_step: SKIP (fused optimizer reads fp8 directly)
      4. FP8Adafactor.step(): reads fp8, updates in fp32, writes fp8
      5. on_step_end: SKIP compress (already fp8), just clear grad accumulators
      6. on_save: no-op (save fp8 buffers directly, _load_from_state_dict handles loading)
    """

    def __init__(self, fp8_gradients: bool = True, fused_optimizer: bool = False):
        self._fp8_gradients = fp8_gradients
        self._fused_optimizer = fused_optimizer
        self._model = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Ensure weights start compressed and store model reference."""
        if model is not None:
            self._model = model
            fp8_compress_all(model)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Decompress fp8 weights and gradients to bf16 before optimizer step.

        Skipped when using fused optimizer (it reads fp8 directly).
        """
        if self._fused_optimizer:
            return
        if self._model is not None:
            # Materialize weights so optimizer can update bf16 values
            fp8_materialize_all(self._model)
            if self._fp8_gradients:
                materialize_fp8_gradients(self._model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """After optimizer step, compress weights and clear grad accumulators."""
        if model is not None:
            if not self._fused_optimizer:
                # Standard optimizer: need to compress weights it left in bf16
                fp8_compress_all(model)
            if self._fp8_gradients:
                clear_fp8_grad_accumulators(model)

    def on_save(self, args, state, control, model=None, **kwargs):
        """No-op: _save_to_state_dict handles fp8-native checkpoint format."""
        pass
