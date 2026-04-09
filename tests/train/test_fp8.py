import pytest
import torch
import torch.nn as nn
from llamafactory.train.fp8_linear import FP8StorageLinear, quantize_to_fp8, dequantize_from_fp8
from llamafactory.train.fp8_pure import FP8PureLinear, _check_native_fp8_support

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_quantize_dequantize():
    device = torch.device('cuda:0')
    tensor = torch.randn(128, 128, dtype=torch.bfloat16, device=device)
    
    fp8_tensor, scale = quantize_to_fp8(tensor, dtype=torch.float8_e4m3fn)
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.bfloat16
    
    dequantized = dequantize_from_fp8(fp8_tensor, scale, dtype=torch.bfloat16)
    assert dequantized.dtype == torch.bfloat16
    assert dequantized.shape == tensor.shape
    
    # Check max error (fp8 has limited precision, so error bound is relatively high)
    max_err = (tensor - dequantized).abs().max().item()
    assert max_err < 0.5  # reasonable bound for e4m3 with random normal data

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fp8_storage_linear():
    device = torch.device('cuda:0')
    linear = nn.Linear(64, 32, bias=True).to(device, dtype=torch.bfloat16)
    
    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=False)
    
    assert hasattr(fp8_linear, "_weight_fp8")
    assert fp8_linear._weight_fp8.dtype == torch.float8_e4m3fn
    assert fp8_linear.weight.numel() == 0  # compressed
    
    # Forward pass
    input_tensor = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
    output = fp8_linear(input_tensor)
    
    assert output.shape == (2, 32)
    assert output.dtype == torch.bfloat16

@pytest.mark.skipif(not torch.cuda.is_available() or not _check_native_fp8_support(), reason="Requires native FP8 support (sm_89+)")
def test_fp8_pure_linear():
    device = torch.device('cuda:0')
    linear = nn.Linear(64, 32, bias=True).to(device, dtype=torch.bfloat16)
    
    fp8_pure = FP8PureLinear.from_linear(linear)
    
    assert fp8_pure.weight.dtype == torch.bfloat16
    assert fp8_pure.weight.shape == (32, 64)
    
    # Forward pass
    input_tensor = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
    output = fp8_pure(input_tensor)
    
    assert output.shape == (2, 32)
    assert output.dtype == torch.bfloat16

@pytest.mark.skipif(not torch.cuda.is_available() or not _check_native_fp8_support(), reason="Requires native FP8 support (sm_89+)")
def test_fp8_storage_linear_native():
    device = torch.device('cuda:0')
    linear = nn.Linear(64, 32, bias=True).to(device, dtype=torch.bfloat16)
    
    fp8_linear = FP8StorageLinear.from_linear(linear, use_native_fp8=True)
    
    # Forward pass uses native fp8 _scaled_mm
    input_tensor = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
    output = fp8_linear(input_tensor)
    
    assert output.shape == (2, 32)
    assert output.dtype == torch.bfloat16
    assert fp8_linear._compressed == True  # shouldn't be decompressed
