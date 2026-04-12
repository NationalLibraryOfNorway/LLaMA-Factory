import pytest
import torch
import torch.nn as nn
from llamafactory.train.fp8_linear import (
    FP8StorageLinear,
    FP8StorageExperts,
    FP8StorageCallback,
    quantize_to_fp8,
    dequantize_from_fp8,
    install_fp8_grad_hooks,
    materialize_fp8_gradients,
    clear_fp8_grad_accumulators,
    fp8_compress_all,
    fp8_materialize_all,
)
from llamafactory.train.fp8_pure import _check_native_fp8_support


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_quantize_dequantize():
    device = torch.device("cuda:0")
    tensor = torch.randn(128, 128, dtype=torch.bfloat16, device=device)

    fp8_tensor, scale = quantize_to_fp8(tensor, dtype=torch.float8_e4m3fn)
    assert fp8_tensor.dtype == torch.float8_e4m3fn

    dequantized = dequantize_from_fp8(fp8_tensor, scale, dtype=torch.bfloat16)
    assert dequantized.dtype == torch.bfloat16
    assert dequantized.shape == tensor.shape

    max_err = (tensor - dequantized).abs().max().item()
    assert max_err < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_quantize_dequantize_e5m2():
    """Test gradient-type quantization (float8_e5m2)."""
    device = torch.device("cuda:0")
    tensor = torch.randn(64, 64, dtype=torch.bfloat16, device=device) * 10

    fp8_tensor, scale = quantize_to_fp8(tensor, dtype=torch.float8_e5m2)
    assert fp8_tensor.dtype == torch.float8_e5m2

    dequantized = dequantize_from_fp8(fp8_tensor, scale, dtype=torch.bfloat16)
    assert dequantized.shape == tensor.shape
    max_err = (tensor - dequantized).abs().max().item()
    assert max_err < 5.0  # e5m2 has wider range but less precision


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_storage_linear():
    device = torch.device("cuda:0")
    linear = nn.Linear(64, 32, bias=True).to(device, dtype=torch.bfloat16)

    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=False)

    assert hasattr(fp8_linear, "_weight_fp8")
    assert fp8_linear._weight_fp8.dtype == torch.float8_e4m3fn
    assert fp8_linear.weight.numel() == 0  # compressed
    assert fp8_linear._compressed is True

    # Forward pass
    input_tensor = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
    output = fp8_linear(input_tensor)

    assert output.shape == (2, 32)
    assert output.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_storage_compress_materialize_roundtrip():
    """Test compress -> materialize preserves weights within fp8 tolerance."""
    device = torch.device("cuda:0")
    linear = nn.Linear(64, 32, bias=False).to(device, dtype=torch.bfloat16)
    original_weight = linear.weight.data.clone()

    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=False)
    assert fp8_linear._compressed is True
    assert fp8_linear.weight.numel() == 0

    fp8_linear.materialize()
    assert fp8_linear._compressed is False
    assert fp8_linear.weight.shape == (32, 64)

    max_err = (original_weight - fp8_linear.weight.data).abs().max().item()
    assert max_err < 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_storage_gradient_flow():
    """Test that gradients flow correctly through FP8StorageLinear."""
    device = torch.device("cuda:0")
    linear = nn.Linear(64, 32, bias=True).to(device, dtype=torch.bfloat16)
    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=False)

    input_tensor = torch.randn(2, 64, dtype=torch.bfloat16, device=device, requires_grad=True)
    output = fp8_linear(input_tensor)
    loss = output.sum()
    loss.backward()

    assert input_tensor.grad is not None
    assert input_tensor.grad.shape == (2, 64)
    # Weight grad should exist (optimizer needs it)
    assert fp8_linear.weight.grad is not None or hasattr(fp8_linear.weight, "_grad_fp8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_gradient_compression_hooks():
    """Test that gradient compression hooks compress grads to fp8_e5m2."""
    device = torch.device("cuda:0")
    # Build FP8StorageLinear manually (without from_linear's backward hook)
    # to isolate grad compression testing from the compress-after-backward lifecycle
    fp8_linear = FP8StorageLinear(64, 32, bias=False, device=device, dtype=torch.bfloat16)
    fp8_linear.weight = nn.Parameter(
        torch.randn(32, 64, dtype=torch.bfloat16, device=device)
    )

    # Install grad hooks
    handles = install_fp8_grad_hooks(
        nn.ModuleList([fp8_linear])
    )
    assert len(handles) > 0

    # Forward + backward
    input_tensor = torch.randn(4, 64, dtype=torch.bfloat16, device=device)
    output = fp8_linear(input_tensor)
    loss = output.sum()
    loss.backward()

    # Grad should be compressed: param.grad is None, _grad_fp8 exists
    param = fp8_linear.weight
    assert param.grad is None  # bf16 grad freed by hook
    assert hasattr(param, "_grad_fp8")
    assert param._grad_fp8 is not None
    assert param._grad_fp8.dtype == torch.float8_e5m2

    # Materialize grads for optimizer
    materialize_fp8_gradients(nn.ModuleList([fp8_linear]))
    assert param.grad is not None
    assert param.grad.dtype == torch.bfloat16

    # Clean up
    clear_fp8_grad_accumulators(nn.ModuleList([fp8_linear]))
    assert param._grad_fp8 is None

    for h in handles:
        h.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_callback_lifecycle():
    """Test FP8StorageCallback manages compress/materialize lifecycle."""
    device = torch.device("cuda:0")
    linear = nn.Linear(64, 32, bias=False).to(device, dtype=torch.bfloat16)
    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=False)
    model = nn.ModuleList([fp8_linear])

    callback = FP8StorageCallback(fp8_gradients=False, fused_optimizer=False)

    # on_train_begin: should compress
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    assert fp8_linear._compressed is True

    # on_pre_optimizer_step: should materialize
    callback.on_pre_optimizer_step(args=None, state=None, control=None)
    assert fp8_linear._compressed is False

    # on_step_end: should re-compress
    callback.on_step_end(args=None, state=None, control=None, model=model)
    assert fp8_linear._compressed is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_checkpoint_roundtrip():
    """Test save and load fp8 checkpoint."""
    device = torch.device("cuda:0")
    linear = nn.Linear(64, 32, bias=True).to(device, dtype=torch.bfloat16)
    original_weight = linear.weight.data.clone()
    original_bias = linear.bias.data.clone()

    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=False)

    # Save state dict
    state_dict = fp8_linear.state_dict()
    assert "weight" in state_dict
    assert state_dict["weight"].dtype == torch.float8_e4m3fn  # saved as fp8
    assert "weight_scale" in state_dict

    # Load into new module
    fp8_linear2 = FP8StorageLinear(64, 32, bias=True, device=device, dtype=torch.bfloat16)
    fp8_linear2.load_state_dict(state_dict)
    assert fp8_linear2._compressed is True

    # Materialize and compare
    fp8_linear2.materialize()
    max_weight_err = (original_weight - fp8_linear2.weight.data).abs().max().item()
    assert max_weight_err < 0.5
    assert torch.allclose(original_bias, fp8_linear2.bias.data)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _check_native_fp8_support(),
    reason="Requires native FP8 support (sm_89+)",
)
def test_fp8_storage_linear_native():
    device = torch.device("cuda:0")
    linear = nn.Linear(64, 32, bias=True).to(device, dtype=torch.bfloat16)

    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=True)

    # Forward pass uses native fp8 _scaled_mm
    input_tensor = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
    output = fp8_linear(input_tensor)

    assert output.shape == (2, 32)
    assert output.dtype == torch.bfloat16
    assert fp8_linear._compressed is True  # native path doesn't materialize


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _check_native_fp8_support(),
    reason="Requires native FP8 support (sm_89+)",
)
def test_fp8_native_gradient_flow():
    """Test gradient flow through native fp8 matmul path."""
    device = torch.device("cuda:0")
    linear = nn.Linear(64, 32, bias=True).to(device, dtype=torch.bfloat16)
    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=True)

    # Batch size must be >= 16 for torch._scaled_mm dimension alignment
    input_tensor = torch.randn(16, 64, dtype=torch.bfloat16, device=device, requires_grad=True)
    output = fp8_linear(input_tensor)
    loss = output.sum()
    loss.backward()

    assert input_tensor.grad is not None
    assert input_tensor.grad.shape == (16, 64)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _check_native_fp8_support(),
    reason="Requires native FP8 support (sm_89+)",
)
def test_fp8_native_matmul_speedup():
    """Benchmark: native fp8 scaled_mm should be faster than bf16 matmul.

    Compares raw matmul throughput (not our full forward which includes
    on-the-fly input quantization overhead). Uses dimensions representative
    of transformer hidden layers.
    """
    import time
    from llamafactory.train.fp8_pure import _quantize_e4m3

    device = torch.device("cuda:0")
    dim = 4096
    batch = 512
    warmup = 20
    iters = 100

    # Setup: pre-quantized weight and input (simulates steady-state training)
    weight_bf16 = torch.randn(dim, dim, dtype=torch.bfloat16, device=device)
    input_bf16 = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    weight_fp8, weight_scale = _quantize_e4m3(weight_bf16)
    input_fp8, input_scale = _quantize_e4m3(input_bf16)

    # Benchmark bf16 matmul
    for _ in range(warmup):
        _ = torch.mm(input_bf16, weight_bf16.t())
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch.mm(input_bf16, weight_bf16.t())
    torch.cuda.synchronize()
    bf16_time = (time.perf_counter() - t0) / iters

    # Benchmark fp8 scaled_mm
    for _ in range(warmup):
        _ = torch._scaled_mm(
            input_fp8, weight_fp8.t(),
            scale_a=input_scale, scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = torch._scaled_mm(
            input_fp8, weight_fp8.t(),
            scale_a=input_scale, scale_b=weight_scale,
            out_dtype=torch.bfloat16,
        )
    torch.cuda.synchronize()
    fp8_time = (time.perf_counter() - t0) / iters

    speedup = bf16_time / fp8_time
    print(f"\n  bf16 torch.mm:     {bf16_time*1000:.3f} ms")
    print(f"  fp8 _scaled_mm:    {fp8_time*1000:.3f} ms")
    print(f"  speedup:           {speedup:.2f}x")
    print(f"  dims: ({batch},{dim}) @ ({dim},{dim})")

    # Native fp8 matmul should be faster than bf16 at these dimensions
    assert speedup > 1.0, (
        f"Native fp8 matmul was NOT faster than bf16: {speedup:.2f}x "
        f"(bf16={bf16_time*1000:.1f}ms, fp8={fp8_time*1000:.1f}ms)"
    )


# ---------------------------------------------------------------------------
# FP8 Metrics (on_log callback)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_callback_on_log_metrics():
    """on_log() injects memory, layer count, and quantization error metrics."""
    device = torch.device("cuda:0")
    # Use larger layer so memory savings round to > 0 MB
    linear = nn.Linear(1024, 1024, bias=False).to(device, dtype=torch.bfloat16)
    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=False)
    model = nn.ModuleList([fp8_linear])

    callback = FP8StorageCallback(fp8_gradients=True, fused_optimizer=False)
    callback.on_train_begin(args=None, state=None, control=None, model=model)

    logs = {}
    callback.on_log(args=None, state=None, control=None, logs=logs)

    assert logs["fp8_layers_converted"] == 1
    assert logs["fp8_memory_saved_mb"] > 0
    assert "fp8_weight_quant_error" in logs
    assert 0 <= logs["fp8_weight_quant_error"] < 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_callback_on_log_no_gradients():
    """on_log() reports less memory saved when fp8_gradients=False."""
    device = torch.device("cuda:0")
    linear = nn.Linear(256, 128, bias=False).to(device, dtype=torch.bfloat16)
    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=False)
    model = nn.ModuleList([fp8_linear])

    cb_with = FP8StorageCallback(fp8_gradients=True)
    cb_with.on_train_begin(args=None, state=None, control=None, model=model)
    logs_with = {}
    cb_with.on_log(args=None, state=None, control=None, logs=logs_with)

    cb_without = FP8StorageCallback(fp8_gradients=False)
    cb_without.on_train_begin(args=None, state=None, control=None, model=model)
    logs_without = {}
    cb_without.on_log(args=None, state=None, control=None, logs=logs_without)

    assert logs_with["fp8_memory_saved_mb"] > logs_without["fp8_memory_saved_mb"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_callback_on_log_none_logs():
    """on_log() handles None logs without raising."""
    device = torch.device("cuda:0")
    linear = nn.Linear(64, 32, bias=False).to(device, dtype=torch.bfloat16)
    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=False)
    model = nn.ModuleList([fp8_linear])

    callback = FP8StorageCallback(fp8_gradients=True)
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    callback.on_log(args=None, state=None, control=None, logs=None)  # should not raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_callback_on_log_no_fp8_layers():
    """on_log() reports zero metrics when model has no FP8 layers."""
    model = nn.ModuleList([nn.Linear(64, 32)])

    callback = FP8StorageCallback(fp8_gradients=True)
    callback.on_train_begin(args=None, state=None, control=None, model=model)

    logs = {}
    callback.on_log(args=None, state=None, control=None, logs=logs)

    assert logs["fp8_layers_converted"] == 0
    assert logs["fp8_memory_saved_mb"] == 0.0
    assert "fp8_weight_quant_error" not in logs


# ---------------------------------------------------------------------------
# FP8 + Attention Backend Integration Tests
# ---------------------------------------------------------------------------


class _MiniAttentionBlock(nn.Module):
    """Minimal multi-head attention block with separate Q/K/V/O projections.

    Mirrors the structure of transformer attention layers (LlamaAttention, etc.)
    to test FP8 linear replacement in a realistic attention context.
    """

    def __init__(self, hidden_size: int = 256, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, use_sdpa: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if use_sdpa:
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Eager attention: manual softmax + matmul
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            # Causal mask
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn_out)


class _MiniTransformer(nn.Module):
    """Tiny transformer with attention + FFN for FP8 integration testing."""

    def __init__(self, hidden_size: int = 256, num_heads: int = 4):
        super().__init__()
        self.attn = _MiniAttentionBlock(hidden_size, num_heads)
        self.ffn_up = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.ffn_down = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, use_sdpa: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm(x), use_sdpa=use_sdpa)
        x = x + self.ffn_down(torch.nn.functional.silu(self.ffn_up(self.norm(x))))
        return x


from llamafactory.train.fp8_linear import convert_model_to_fp8_storage


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_with_eager_attention():
    """FP8 linear layers work correctly with eager (manual) attention."""
    device = torch.device("cuda:0")
    model = _MiniTransformer(hidden_size=256, num_heads=4).to(device, dtype=torch.bfloat16)

    # Convert to FP8
    model = convert_model_to_fp8_storage(model, fp8_gradients=False, fp8_mode="storage")

    # Verify projections were converted
    assert isinstance(model.attn.q_proj, FP8StorageLinear)
    assert isinstance(model.attn.k_proj, FP8StorageLinear)
    assert isinstance(model.attn.v_proj, FP8StorageLinear)
    assert isinstance(model.attn.o_proj, FP8StorageLinear)

    # Forward + backward with eager attention
    x = torch.randn(2, 32, 256, dtype=torch.bfloat16, device=device, requires_grad=True)
    out = model(x, use_sdpa=False)
    assert out.shape == (2, 32, 256)
    assert out.dtype == torch.bfloat16

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_with_sdpa_attention():
    """FP8 linear layers work correctly with SDPA attention."""
    device = torch.device("cuda:0")
    model = _MiniTransformer(hidden_size=256, num_heads=4).to(device, dtype=torch.bfloat16)

    # Convert to FP8
    model = convert_model_to_fp8_storage(model, fp8_gradients=False, fp8_mode="storage")

    # Forward + backward with SDPA
    x = torch.randn(2, 32, 256, dtype=torch.bfloat16, device=device, requires_grad=True)
    out = model(x, use_sdpa=True)
    assert out.shape == (2, 32, 256)
    assert out.dtype == torch.bfloat16

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_attention_output_consistency():
    """FP8 eager and SDPA attention produce similar outputs (within fp8 tolerance)."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    # Two identical models
    model_eager = _MiniTransformer(hidden_size=256, num_heads=4).to(device, dtype=torch.bfloat16)
    model_sdpa = _MiniTransformer(hidden_size=256, num_heads=4).to(device, dtype=torch.bfloat16)
    model_sdpa.load_state_dict(model_eager.state_dict())

    # Convert both to FP8
    model_eager = convert_model_to_fp8_storage(model_eager, fp8_gradients=False, fp8_mode="storage")
    model_sdpa = convert_model_to_fp8_storage(model_sdpa, fp8_gradients=False, fp8_mode="storage")

    x = torch.randn(2, 16, 256, dtype=torch.bfloat16, device=device)
    out_eager = model_eager(x, use_sdpa=False)
    out_sdpa = model_sdpa(x, use_sdpa=True)

    # Outputs should be close (same weights, same input, different attention impl)
    max_diff = (out_eager - out_sdpa).abs().max().item()
    assert max_diff < 0.1, f"Eager vs SDPA output diff too large: {max_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_attention_with_gradient_checkpointing():
    """FP8 + SDPA works correctly with gradient checkpointing enabled."""
    device = torch.device("cuda:0")
    model = _MiniTransformer(hidden_size=256, num_heads=4).to(device, dtype=torch.bfloat16)
    model = convert_model_to_fp8_storage(model, fp8_gradients=False, fp8_mode="storage")

    # Simulate gradient checkpointing: forward under no_grad, then backward re-run
    x = torch.randn(2, 16, 256, dtype=torch.bfloat16, device=device, requires_grad=True)

    # First forward under no_grad (gradient checkpointing discards activations)
    model.train()
    with torch.no_grad():
        _ = model(x, use_sdpa=True)

    # Verify projections got re-compressed after no_grad forward
    assert model.attn.q_proj._compressed is True

    # Second forward with grad (backward re-run materializes weights)
    out = model(x, use_sdpa=True)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _check_native_fp8_support(),
    reason="Requires native FP8 support (sm_89+)",
)
def test_fp8_native_with_sdpa_attention():
    """Native FP8 matmul path works with SDPA attention."""
    device = torch.device("cuda:0")
    model = _MiniTransformer(hidden_size=256, num_heads=4).to(device, dtype=torch.bfloat16)
    model = convert_model_to_fp8_storage(model, fp8_gradients=False, fp8_mode="native")

    # All projections should use native fp8
    assert model.attn.q_proj._use_native_fp8 is True

    x = torch.randn(2, 16, 256, dtype=torch.bfloat16, device=device, requires_grad=True)
    out = model(x, use_sdpa=True)
    assert out.shape == (2, 16, 256)
    assert out.dtype == torch.bfloat16

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_attention_full_training_step():
    """Full training step: FP8 + SDPA + optimizer + callback lifecycle."""
    device = torch.device("cuda:0")
    model = _MiniTransformer(hidden_size=256, num_heads=4).to(device, dtype=torch.bfloat16)
    model = convert_model_to_fp8_storage(model, fp8_gradients=True, fp8_mode="storage")

    callback = FP8StorageCallback(fp8_gradients=True, fused_optimizer=False)
    callback.on_train_begin(args=None, state=None, control=None, model=model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(3):
        x = torch.randn(2, 16, 256, dtype=torch.bfloat16, device=device)
        out = model(x, use_sdpa=True)
        loss = out.sum()
        loss.backward()

        # Pre-optimizer: materialize weights + grads
        callback.on_pre_optimizer_step(args=None, state=None, control=None)

        optimizer.step()
        optimizer.zero_grad()

        # Post-step: re-compress
        callback.on_step_end(args=None, state=None, control=None, model=model)

        # Verify model is back in compressed state
        assert model.attn.q_proj._compressed is True

    # Final output should be finite
    x = torch.randn(2, 16, 256, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        out = model(x, use_sdpa=True)
    assert torch.isfinite(out).all()
