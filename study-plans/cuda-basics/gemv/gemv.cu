#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
    extern __shared__ float s_x[];

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        s_x[i] = x[i];
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += A[row * N + col] * s_x[col];
        }
        y[row] = sum;
    }
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    int threads = THREADS_PER_BLOCK;
    int blocks = (M + threads - 1) / threads;
    size_t shared_mem_bytes = N * sizeof(float);

    gemv_kernel<<<blocks, threads, shared_mem_bytes>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}