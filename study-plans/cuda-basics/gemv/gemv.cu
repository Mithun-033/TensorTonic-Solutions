#include <cuda_runtime.h>

#define BLOCK_SIZE_M 8
#define BLOCK_SIZE_N 32

__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
    extern __shared__ float s_x[];
    __shared__ float s_sum[BLOCK_SIZE_M][BLOCK_SIZE_N];

    int tid = threadIdx.y * BLOCK_SIZE_N + threadIdx.x;
    int num_threads = BLOCK_SIZE_M * BLOCK_SIZE_N;

    for (int i = tid; i < N; i += num_threads) {
        s_x[i] = x[i];
    }
    __syncthreads();

    int row = blockIdx.x * BLOCK_SIZE_M + threadIdx.y;
    float sum = 0.0f;

    if (row < M) {
        for (int col = threadIdx.x; col < N; col += BLOCK_SIZE_N) {
            sum += A[row * N + col] * s_x[col];
        }
    }

    s_sum[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE_N / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.y][threadIdx.x] += s_sum[threadIdx.y][threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && row < M) {
        y[row] = s_sum[threadIdx.y][0];
    }
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    dim3 threads(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 blocks((M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    size_t shared_mem_bytes = N * sizeof(float);
    gemv_kernel<<<blocks, threads, shared_mem_bytes>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}