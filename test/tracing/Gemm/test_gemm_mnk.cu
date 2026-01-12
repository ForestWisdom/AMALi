#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)

constexpr int TILE = 16;

// C[m x n] = A[m x k] * B[k x n], row-major
__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int m, int n, int k)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    // loop over tiles of k dimension
    for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x; // in [0, k)
        int b_row = t * TILE + threadIdx.y; // in [0, k)

        // load A tile
        if (row < m && a_col < k) As[threadIdx.y][threadIdx.x] = A[row * k + a_col];
        else As[threadIdx.y][threadIdx.x] = 0.0f;

        // load B tile
        if (b_row < k && col < n) Bs[threadIdx.y][threadIdx.x] = B[b_row * n + col];
        else Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = acc;
    }
}

static void fill(float* h, size_t n, float v) {
    for (size_t i = 0; i < n; ++i) h[i] = v;
}

static void run_one(int m, int n, int k, const char* tag) {
    size_t bytesA = (size_t)m * k * sizeof(float);
    size_t bytesB = (size_t)k * n * sizeof(float);
    size_t bytesC = (size_t)m * n * sizeof(float);

    float *hA = (float*)std::malloc(bytesA);
    float *hB = (float*)std::malloc(bytesB);
    float *hC = (float*)std::malloc(bytesC);
    if (!hA || !hB || !hC) { fprintf(stderr, "Host malloc failed\n"); std::exit(1); }

    // 简单初始化：全 1，方便 sanity check（C 每个元素应为 k）
    fill(hA, (size_t)m * k, 1.0f);
    fill(hB, (size_t)k * n, 1.0f);

    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    CK(cudaMalloc(&dA, bytesA));
    CK(cudaMalloc(&dB, bytesB));
    CK(cudaMalloc(&dC, bytesC));

    CK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

    // 计时（可选）
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));
    CK(cudaEventRecord(start));

    gemm_tiled<<<grid, block>>>(dA, dB, dC, m, n, k);
    CK(cudaGetLastError());
    CK(cudaEventRecord(stop));
    CK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, start, stop));

    CK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

    // 简单校验：抽样检查几个点 ~ k（允许小误差）
    float c0 = hC[0];
    float c1 = hC[(size_t)(m/2) * n + (n/2)];
    printf("[%s] m=%d n=%d k=%d | mnk=%lld | time=%.3f ms | C[0]=%.1f C[mid]=%.1f (expect ~%d)\n",
           tag, m, n, k,
           (long long)m * n * k, ms, c0, c1, k);

    CK(cudaEventDestroy(start));
    CK(cudaEventDestroy(stop));

    CK(cudaFree(dA));
    CK(cudaFree(dB));
    CK(cudaFree(dC));
    std::free(hA);
    std::free(hB);
    std::free(hC);
}

int main() {
    // 两次发射：同名 kernel，保证 mnk 相同但 (m,n,k) 不同
    // A: 更方
    run_one(1024, 1024, 256, "shape-A(k=256, 1024x1024 output)");
    // run_one(2048, 2048, 64,  "shape-B(k=64, 2048x2048 output)");

    // run_one(2048, 2048, 64,  "shape-B(k=64, 2048x2048 output)");
    // run_one(1024, 1024, 256, "shape-A(k=256, 1024x1024 output)");

    CK(cudaDeviceSynchronize());
    return 0;
}
