#include <cuda_runtime.h>

#define BLOCK_SIZE_M 8
#define BLOCK_SIZE_N 32

__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
    extern __shared__ float s_x[];

    int tid = threadIdx.y * BLOCK_SIZE_N + threadIdx.x;
    int num_threads = BLOCK_SIZE_M * BLOCK_SIZE_N;

    for (int i = tid; i < N; i += num_threads) {
        s_x[i] = x[i];
    }
    __syncthreads();

    int row = blockIdx.x * BLOCK_SIZE_M + threadIdx.y;

    if (row < M) {
        float sum = 0.0f;

        #pragma unroll 4
        for (int col = threadIdx.x; col < N; col += BLOCK_SIZE_N) {
            sum += A[row * N + col] * s_x[col];
        }

        #pragma unroll
        for (int stride = BLOCK_SIZE_N >> 1; stride > 0; stride >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, stride);
        }

        if (threadIdx.x == 0) {
            y[row] = sum;
        }
    }
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    dim3 threads(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 blocks((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    size_t shared_mem_bytes = N * sizeof(float);
    gemv_kernel<<<blocks, threads, shared_mem_bytes>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}