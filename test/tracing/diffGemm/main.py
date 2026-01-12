#!/usr/bin/env python3
import argparse
import csv
import json
import torch

from gemm_half import get_available_backends
from gemm_half import benchmark_gemm as bench_gemm_16bit  # 同 main.py 用法一致

def parse_args():
    p = argparse.ArgumentParser("GEMM benchmark (no bench_gpu_time, no multiprocessing)")
    p.add_argument("--csv", type=str, default=None,
                   help="CSV file with M,N,K configs (columns: m,n,k)")
    p.add_argument("--sizes", nargs="+", type=int, default=[1024],
                   help="If no --csv, use these square sizes (M=N=K=size)")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"],
                   help="Datatype")
    p.add_argument("--backends", nargs="+", default=None,
                   help="Backends to test (e.g., torch triton). If omitted, auto-detect.")
    p.add_argument("--num-warmup", type=int, default=1)
    p.add_argument("--num-iterations", type=int, default=1)

    # 单 GPU 选择：避免 device_count>1 时搞并行
    p.add_argument("--gpu-id", type=int, default=0,
                   help="Which cuda device id to use (default 0)")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda or cpu")
    p.add_argument("--output", type=str, default="gemm_results_no_mp.json")
    return p.parse_args()

def load_configurations(args):
    configs = []
    if args.csv:
        with open(args.csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m = int(row["m"]); n = int(row["n"]); k = int(row["k"])
                configs.append((m, n, k))
    else:
        for s in args.sizes:
            configs.append((s, s, s))
    return configs

def main():
    args = parse_args()

    # dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # device（强制单 GPU）
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        assert torch.cuda.is_available(), "CUDA not available"
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")

    print(f"Running on device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    configs = load_configurations(args)
    print(f"Loaded {len(configs)} configurations")

    results = []

    for (M, N, K) in configs:
        print(f"\nBenchmarking {M}x{N}x{K}")

        backends = get_available_backends(dtype)
        if args.backends:
            backends = [b for b in backends if b.name in args.backends]

        for backend in backends:
            try:
                print(f"  Running backend={backend.name} dtype={args.dtype} ...")

                # 关键：强制 use_advanced_timing=False，
                # 这样会绕开 bench_gpu_time 分支，走简单计时路径（warmup + time.time + synchronize）
                result = bench_gemm_16bit(
                    backend, M, N, K,
                    num_warmup=args.num_warmup,
                    num_iterations=args.num_iterations,
                    device=device,
                    use_advanced_timing=False,
                )

                results.append(result)
                print(f"    avg_time_ms={result['avg_time_ms']:.4f}  tflops={result['tflops']:.3f}")
            except Exception as e:
                print(f"    ERROR: {e}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.output}")

if __name__ == "__main__":
    # 不要 set_start_method，不要 multiprocessing
    main()
