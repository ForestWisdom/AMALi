#!/usr/bin/env python3
"""
GEMM Half Precision Backend Implementations

This module contains GEMM backend classes and benchmarking functions
for half-precision operations (FP16, BF16).
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

# Import SGLang components for device detection
try:
    from sglang.srt.utils import (
        get_device_capability,
        get_device_sm,
        is_cuda,
        is_hip,
        is_sm90_supported,
        is_sm100_supported,
    )
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    warnings.warn("SGLang not available, some utilities will be disabled")

# Try to import optional backends
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    from flashinfer.testing.utils import bench_gpu_time
    FLASHINFER_TESTING_AVAILABLE = True
except ImportError:
    FLASHINFER_TESTING_AVAILABLE = False

import numpy as np


class GEMMBackend:
    """Base class for GEMM backends."""

    def __init__(self, name: str, dtype: torch.dtype):
        self.name = name
        self.dtype = dtype

    def is_supported(self) -> bool:
        """Check if this backend is supported on current hardware."""
        return True

    def prepare_weights(self, M: int, N: int, K: int, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare weight matrix and optional scale."""
        raise NotImplementedError

    def prepare_input(self, M: int, K: int, device: torch.device) -> torch.Tensor:
        """Prepare input matrix."""
        raise NotImplementedError

    def gemm(self, input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform GEMM operation."""
        raise NotImplementedError

    def get_flops(self, M: int, N: int, K: int) -> int:
        """Calculate theoretical FLOPs for M x K @ K x N."""
        return 2 * M * N * K  # Multiply-add operations


class TorchBackend(GEMMBackend):
    """Standard PyTorch/cuBLAS backend."""

    def __init__(self, dtype: torch.dtype):
        super().__init__("torch", dtype)

    # def prepare_weights(self, M: int, N: int, K: int, device: torch.device):
    #     return torch.randn(N, K, dtype=self.dtype, device=device), None

    # def prepare_input(self, M: int, K: int, device: torch.device):
    #     return torch.randn(M, K, dtype=self.dtype, device=device)
    def prepare_weights(self, M: int, N: int, K: int, device: torch.device):
        return torch.randn(N, K, dtype=self.dtype, device="cpu").cuda(), None

    def prepare_input(self, M: int, K: int, device: torch.device):
        return torch.randn(M, K, dtype=self.dtype, device="cpu").cuda()

    def gemm(self, input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        return F.linear(input_tensor, weight, bias)


class TritonBackend(GEMMBackend):
    """Triton GEMM backend."""

    def __init__(self, dtype: torch.dtype):
        super().__init__("triton", dtype)

    def is_supported(self) -> bool:
        return TRITON_AVAILABLE and is_cuda()

    def prepare_weights(self, M: int, N: int, K: int, device: torch.device):
        return torch.randn(N, K, dtype=self.dtype, device=device), None

    def prepare_input(self, M: int, K: int, device: torch.device):
        return torch.randn(M, K, dtype=self.dtype, device=device)

    def gemm(self, input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        # Use a simple triton matmul for demonstration
        # In practice, you'd use optimized kernels
        return torch.matmul(input_tensor, weight.t())


def benchmark_gemm(
    backend: GEMMBackend,
    M: int,
    N: int,
    K: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: torch.device = None,
    use_advanced_timing: bool = False
) -> Dict[str, Any]:
    """Benchmark a single GEMM operation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get device name
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = str(device)

    # Prepare inputs
    weight, weight_scale = backend.prepare_weights(M, N, K, device)
    input_tensor = backend.prepare_input(M, K, device)
    bias = None  # Could add bias support

    # Define the GEMM function
    def gemm_func():
        return backend.gemm(input_tensor, weight, bias)

    # Initialize avg_time
    avg_time = None

    # Use advanced timing if available and requested
    if use_advanced_timing and device.type == "cuda" and FLASHINFER_TESTING_AVAILABLE:
        try:
            # Use FlashInfer's bench_gpu_time for more accurate measurement
            measurements = bench_gpu_time(
                gemm_func,
                dry_run_time_ms=25,
                repeat_time_ms=100,  # 100ms should be enough for most GEMM operations
                use_cuda_graph=True,
                enable_cupti=True,
            )
            avg_time = float(np.median(measurements)) / 1000  # Convert ms to seconds
        except Exception:
            # Fallback to simple timing
            use_advanced_timing = False
    else:
        use_advanced_timing = False

    if avg_time is None:
        # Simple timing method
        # Warmup
        for _ in range(num_warmup):
            _ = gemm_func()
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = gemm_func()

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_iterations

    # Calculate metrics
    flops = backend.get_flops(M, N, K)
    tflops = flops / (avg_time * 1e12)  # TFLOPS

    return {
        "backend": backend.name,
        "dtype": str(backend.dtype),
        "M": M,
        "N": N,
        "K": K,
        "avg_time_ms": avg_time * 1000,
        "tflops": tflops,
        "total_flops": flops,
        "device": device_name
    }


def get_available_backends(dtype: torch.dtype) -> List[GEMMBackend]:
    """Get list of available backends for given dtype."""
    backends = []

    # Always add torch backend
    backends.append(TorchBackend(dtype))

    if TritonBackend(dtype).is_supported():
        backends.append(TritonBackend(dtype))

    return [b for b in backends if b.is_supported()]
