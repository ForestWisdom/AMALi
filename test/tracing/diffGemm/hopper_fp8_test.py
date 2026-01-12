import copy
import numpy as np
import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    ignore_env, get_arch_major
)

from DeepGEMM.tests.generators import (
    KernelType, get_ue8m0_usage,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous
)


def test_gemm(m, n, k, num_warmup=10, num_iterations=1) -> dict:
    # Use the first enumerated config as template for other parameters
    from DeepGEMM.tests.generators import KernelType
    kernel_type = KernelType.Kernel1D1D
    major_a = major_b = accumulate = out_dtype = None  # Will be set by generate_normal if needed
    first_config = next(enumerate_normal(torch.float8_e4m3fn))
    _, _, _, _, major_a, major_b, accumulate, out_dtype = first_config
    test_configs = [(kernel_type, m, n, k, major_a, major_b, accumulate, out_dtype)]

    kernel_type, m, n, k, major_a, major_b, accumulate, out_dtype = test_configs[0]
    major_opt  = 'N' if major_a.is_k_major() else 'T'
    major_opt += 'T' if major_b.is_k_major() else 'N'
    out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
    acc_opt    = f'acc={int(accumulate)}'
    kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
    use_ue8m0 = get_ue8m0_usage(kernel_type)
    disable_ue8m0_cast = not use_ue8m0
    recipe = (1, 1, 128) if kernel_type.is_1d1d() and accumulate else None

    for test_alias in (False, True):
        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0)
        func_name = f'fp8_gemm_{major_opt.lower() if test_alias else "nt"}'
        if test_alias:
            a = a if major_a.is_k_major() else (a[0].T, a[1].T)
            b = b if major_b.is_k_major() else (b[0].T, b[1].T)
            assert a[0].is_contiguous() and b[0].is_contiguous()
        getattr(deep_gemm, func_name)(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe)
        diff = calc_diff(d, ref_d)
        assert diff < 0.001, (f'{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                              f'{diff:.5f}, alias={test_alias}')

    a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0)
    for _ in range(num_warmup):
        deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe)
        torch.cuda.synchronize()
    t = bench_kineto(lambda: [deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe) for _ in range(num_iterations)],
                     'fp8_gemm', suppress_kineto_output=True)
    for _ in range(num_warmup):
        deep_gemm.cublaslt_gemm_nt(a[0], b[0], d, c=c)
        torch.cuda.synchronize()
    cublas_t, split_k_t = bench_kineto(lambda: [deep_gemm.cublaslt_gemm_nt(a[0], b[0], d, c=c) for _ in range(num_iterations)][-1], ('nvjet', 'reduce'), suppress_kineto_output=True)

    result = {
        "config": {
            "m": m,
            "n": n,
            "k": k,
            "kernel_opt": kernel_opt,
            "layout": major_opt,
            "out_opt": out_opt,
            "acc_opt": acc_opt
        },
        "backends": []
    }

    if t > 0:
        deepgemm_result = {
            "name": "DeepGEMM",
            "tflops": 2 * m * n * k / t / 1e12,
            "time_us": t * 1e6
        }
        result["backends"].append(deepgemm_result)

    if cublas_t > 0:
        cublas_total_t = cublas_t + split_k_t
        if cublas_total_t > 0:
            cublas_result = {
                "name": "cuBLAS",
                "tflops": 2 * m * n * k / cublas_total_t / 1e12,
                "time_us": cublas_total_t * 1e6
            }
            result["backends"].append(cublas_result)

    return result


def test_m_grouped_gemm_contiguous(configurations=None) -> None:
    print('Testing m-grouped contiguous GEMM:')

    test_configs = []
    for kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        if configurations is None or (expected_m_per_group, n, k) in configurations:
            test_configs.append((kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b))

    for kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b in test_configs:
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        for test_alias in (False, True):
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0)
            func_name = f"m_grouped_fp8_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast)
            d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}, alias={test_alias}'
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked(configurations=None) -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    test_configs = []
    for kernel_type, num_groups, max_m, expected_m_per_group, n, k in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        if configurations is None or (max_m, n, k) in configurations:
            test_configs.append((kernel_type, num_groups, max_m, expected_m_per_group, n, k))

    for kernel_type, num_groups, max_m, expected_m_per_group, n, k in test_configs:
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)
            deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)
            for j in range(num_groups):
                if masked_m[j].item() == 0:
                    continue
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                assert diff < 0.001, f'{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)

        # Test performance with fixed shapes
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous(configurations=None) -> None:
    print('Testing k-grouped contiguous GEMM:')

    k_grouped_fp8_gemm_contiguous = deep_gemm.k_grouped_fp8_gemm_nt_contiguous if get_arch_major() == 9 \
                                    else deep_gemm.k_grouped_fp8_gemm_tn_contiguous

    test_configs = []
    for num_groups, m, n, major_a, major_b, ks, expected_k_per_group in enumerate_k_grouped_contiguous(torch.float8_e4m3fn):
        k = sum(ks)  # Total K dimension
        if configurations is None or (m, n, k) in configurations:
            test_configs.append((num_groups, m, n, major_a, major_b, ks, expected_k_per_group))

    for num_groups, m, n, major_a, major_b, ks, expected_k_per_group in test_configs:
        use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

        for test_empty_groups in (False, True):
            new_ks = copy.deepcopy(ks)
            if test_empty_groups and len(ks) > 1:
                new_ks[random.randint(0, num_groups - 1)] = 0
            k, a, b, c, d, ref_d = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, new_ks, use_ue8m0=use_ue8m0)
            new_ks_tensor = torch.tensor(new_ks, dtype=torch.int, device='cuda')
            k_grouped_fp8_gemm_contiguous(a, b, d, new_ks, new_ks_tensor, c)

            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {k=}, {ks=}, {diff:.5f}'

        # Test performance
        k, a, b, c, d, ref_d = generate_k_grouped_contiguous(num_groups, m, n, major_a, major_b, ks, use_ue8m0=use_ue8m0)
        ks_tensor = torch.tensor(ks, dtype=torch.int, device='cuda')

        # noinspection PyShadowingNames
        def test_func():
            k_grouped_fp8_gemm_contiguous(a, b, d, ks, ks_tensor, c)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=:2}, m={m:5}, n={n:5}, k={k:5}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')
    print()


def run_hopper_fp8_tests(configurations=None):
    """Run all Hopper FP8 tests.

    Args:
        configurations: List of (M, N, K) tuples to test. If None, uses default enumeration.
    """
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm(configurations)
    # test_m_grouped_gemm_contiguous(configurations)
    # test_m_grouped_gemm_masked(configurations)
    # test_k_grouped_gemm_contiguous(configurations)


if __name__ == '__main__':
    run_hopper_fp8_tests()
