#!/usr/bin/env python3
"""
General GEMM Performance Benchmark for SGLang

This script benchmarks GEMM performance across different:
- GPU architectures
- Numerical precisions (FP32, FP16, BF16)
- Input sizes (M, N, K)
- GEMM backends (Torch, Triton)

Also supports running Hopper FP8 tests automatically when fp8 is specified on Hopper GPUs.
"""

import argparse
import csv
import json
import multiprocessing
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np

import torch

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

# Import GEMM half precision backends
from gemm_half import GEMMBackend, TorchBackend, TritonBackend, benchmark_gemm, get_available_backends
from gemm_half import benchmark_gemm as bench_gemm_16bit

# Import Hopper FP8 test module - lazy import to avoid CUDA context pollution
HOPPER_FP8_TEST_AVAILABLE = True
_bench_gemm_fp8_func = None


def run_benchmarks_for_gpu(device_id: int, configurations: List[Tuple[int, int, int]], args: argparse.Namespace, result_queue: multiprocessing.Queue):
    """Run benchmarks for a specific GPU device."""
    # Set device for this process
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    gpu_name = torch.cuda.get_device_name(device)
    sm_version = get_device_sm()
    print(f"Process for GPU {device_id}: {gpu_name} (SM {sm_version})")

    # Map string dtypes to torch dtypes
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
    }
    dtype_str = args.dtype
    dtype = dtype_map[dtype_str]

    results = []

    for M, N, K in configurations:
        print(f"GPU {device_id}: Benchmarking size {M}x{N}x{K}")

        # Special handling for Hopper FP8 tests
        if dtype_str == "fp8" and sm_version >= 90 and HOPPER_FP8_TEST_AVAILABLE:
            # Lazy import to avoid CUDA context pollution
            global _bench_gemm_fp8_func
            if _bench_gemm_fp8_func is None:
                try:
                    from hopper_fp8_test import test_gemm as bench_gemm_fp8_func
                    _bench_gemm_fp8_func = bench_gemm_fp8_func
                except ImportError:
                    print(f"GPU {device_id}: Failed to import FP8 test module")
                    continue
            try:
                gemm_result = _bench_gemm_fp8_func(M, N, K, num_warmup=args.num_warmup, num_iterations=args.num_iterations)
                print(f"GPU {device_id}: {gemm_result}")
                if gemm_result:
                    for backend in gemm_result['backends']:
                        print(f"GPU {device_id}: Running {backend['name']}...")
                        print(f"GPU {device_id}: {backend['name']}: {backend['tflops']:.2f} TFLOPS")
                        result_entry = {
                            "backend": backend['name'],
                            "dtype": "fp8",
                            "M": gemm_result['config']['m'],
                            "N": gemm_result['config']['n'],
                            "K": gemm_result['config']['k'],
                            "avg_time_ms": backend['time_us'] / 1000,
                            "tflops": backend['tflops'],
                            "total_flops": 2 * gemm_result['config']['m'] * gemm_result['config']['n'] * gemm_result['config']['k'],
                            "device": gpu_name
                        }
                        results.append(result_entry)
            except Exception as e:
                print(f"GPU {device_id}: Error with Hopper FP8 test: {e}")
                continue
        else:
            backends = get_available_backends(dtype)
            if args.backends:
                backends = [b for b in backends if b.name in args.backends]

            for backend in backends:
                try:
                    print(f"GPU {device_id}: Running {backend.name}...")
                    result = bench_gemm_16bit(
                        backend, M, N, K,
                        num_warmup=args.num_warmup,
                        num_iterations=args.num_iterations,
                        device=device
                    )
                    results.append(result)
                    print(f"GPU {device_id}: {backend.name}: {result['tflops']:.2f} TFLOPS")
                except Exception as e:
                    print(f"GPU {device_id}: Error with {backend.name}: {e}")
                    continue

    # Put results into queue
    result_queue.put((device_id, results))



def main():
    global _bench_gemm_fp8_func
    parser = argparse.ArgumentParser(description="GEMM Performance Benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1024, 2048, 4096, 8192],
                       help="Matrix sizes to test (square matrices)")
    parser.add_argument("--csv", type=str, default=None,
                       help="CSV file with M,N,K configurations (columns: m,n,k)")
    parser.add_argument("--dtype", default="bf16",
                       help="Data type to test (single precision per execution)")
    parser.add_argument("--backends", nargs="+", default=None,
                       help="Backends to test (auto-detect if not specified)")
    parser.add_argument("--num-warmup", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=100,
                       help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default="gemm_benchmark_results.json",
                       help="Output JSON file")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run on (cuda/cpu)")


    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running benchmarks on device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name()
        sm_version = get_device_sm()
        print(f"GPU: {gpu_name} (SM {sm_version})")
        # Check SM capability for FP8 support
        if sm_version < 89 and args.dtype == "fp8":
            print("Error: FP8 not supported on SM < 89")
            return
    else:
        print("CPU: Running on CPU")

    # Map string dtypes to torch dtypes
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
    }

    if args.dtype not in dtype_map:
        print(f"Error: Unknown dtype {args.dtype}")
        return

    results = []

    # Get configurations to test
    configurations = []
    if args.csv:
        print(f"Reading configurations from {args.csv}")
        with open(args.csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                M = int(row['m'])
                N = int(row['n'])
                K = int(row['k'])
                configurations.append((M, N, K))
        print(f"Loaded {len(configurations)} configurations from CSV")
    else:
        for size in args.sizes:
            M = N = K = size  # Square matrices
            configurations.append((M, N, K))

    # Check for multi-GPU setup
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device.type == "cuda" and num_gpus > 1:
        print(f"Detected {num_gpus} GPUs. Running benchmarks in parallel across GPUs.")
        # Distribute configurations across GPUs
        gpu_configs = [configurations[i::num_gpus] for i in range(num_gpus)]

        result_queue = multiprocessing.Queue()
        processes = []

        for gpu_id in range(num_gpus):
            p = multiprocessing.Process(target=run_benchmarks_for_gpu, args=(gpu_id, gpu_configs[gpu_id], args, result_queue))
            p.start()
            processes.append(p)

        # Collect results from all GPUs
        for _ in range(num_gpus):
            gpu_id, gpu_results = result_queue.get()
            results.extend(gpu_results)

        # Wait for all processes to finish
        for p in processes:
            p.join()
    else:
        # Single GPU or CPU: run sequentially
        if device.type == "cuda":
            device_id = 0  # Assuming cuda:0
        else:
            device_id = -1  # CPU, but we'll handle separately
        print("Running benchmarks sequentially.")

        if device.type == "cuda":
            # Use the function for consistency
            result_queue = multiprocessing.Queue()
            run_benchmarks_for_gpu(0, configurations, args, result_queue)
            _, gpu_results = result_queue.get()
            results.extend(gpu_results)
        else:
            # CPU mode: run original sequential loop
            dtype_str = args.dtype
            dtype = dtype_map[dtype_str]
            print(f"  Testing {dtype_str} on CPU...")
            for M, N, K in configurations:
                print(f"\nBenchmarking size {M}x{N}x{K}")
                backends = get_available_backends(dtype)
                if args.backends:
                    backends = [b for b in backends if b.name in args.backends]

                for backend in backends:
                    try:
                        print(f"    Running {backend.name}...")
                        result = bench_gemm_16bit(
                            backend, M, N, K,
                            num_warmup=args.num_warmup,
                            num_iterations=args.num_iterations,
                            device=device
                        )
                        results.append(result)
                        print(f"      {backend.name}: {result['tflops']:.2f} TFLOPS")
                    except Exception as e:
                        print(f"      Error with {backend.name}: {e}")
                        continue

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\nSummary:")
    for result in results:
        print(f"{result['backend']} {result['dtype']} {result['M']}x{result['N']}x{result['K']}: {result['tflops']:.1f} TFLOPS")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
