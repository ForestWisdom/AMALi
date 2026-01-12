import argparse
import torch

def bench_gemm(m, n, k, dtype, tf32, warmup, iters, tag="gemm"):
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    device = "cuda"
    a = torch.ones((m, k), device=device, dtype=dtype)
    b = torch.ones((k, n), device=device, dtype=dtype)

    # 让 cudnn/cublas 做一些内部初始化（不计时）
    torch.cuda.synchronize()

    # warm-up（不计时）
    for _ in range(warmup):
        c = a @ b
    torch.cuda.synchronize()

    # 正式计时：cuda events（只测 kernel 时间）
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / iters

    # 简单 sanity check：全 1 的话，C 元素 ≈ k
    c0 = float(c[0, 0].detach().cpu())
    cmid = float(c[m // 2, n // 2].detach().cpu())

    print(f"[{tag}] m={m} n={n} k={k} | mnk={m*n*k} | dtype={dtype} tf32={tf32} "
          f"| warmup={warmup} iters={iters} | avg={avg_ms:.6f} ms "
          f"| C[0,0]={c0:.1f} C[mid]={cmid:.1f} (expect ~{k})")

def parse_dtype(s: str):
    s = s.lower()
    if s in ["fp16", "float16"]:
        return torch.float16
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if s in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="fp16", help="fp16|bf16|fp32")
    ap.add_argument("--tf32", type=int, default=1, help="1 enable TF32 (fp32 matmul), 0 disable")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--two_shapes", type=int, default=0, help="1: run two shapes with same mnk")
    args = ap.parse_args()

    dtype = parse_dtype(args.dtype)
    tf32 = bool(args.tf32)

    # 可选：两组 shape（mnk 相同）
    if args.two_shapes:
        # mnk = 268,435,456
        bench_gemm(1024, 1024, 256, dtype, tf32, args.warmup, args.iters, tag="shape-A")
        bench_gemm(2048, 2048, 64,  dtype, tf32, args.warmup, args.iters, tag="shape-B")
    else:
        bench_gemm(args.m, args.n, args.k, dtype, tf32, args.warmup, args.iters, tag="shape")

if __name__ == "__main__":
    main()
