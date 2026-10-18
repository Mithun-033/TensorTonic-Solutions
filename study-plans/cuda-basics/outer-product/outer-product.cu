#include <cuda_runtime.h>

#define TILE_Y 16
#define TILE_X 16

__global__ void outer_product_kernel(const float* a, const float* b, float* C, int M, int N) {
    __shared__ float s_a[TILE_Y];
    __shared__ float s_b[TILE_X];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Threads 0 to 15 load vector 'a'
    if (tid < 16) {
        int g_a_idx = blockIdx.y * blockDim.y + tid;
        s_a[tid] = (g_a_idx < M) ? a[g_a_idx] : 0.0f;
    }
    // Threads 16 to 31 load vector 'b'
    else if (tid < 32) {
        int g_b_idx = blockIdx.x * blockDim.x + (tid - 16);
        s_b[tid - 16] = (g_b_idx < N) ? b[g_b_idx] : 0.0f;
    }

    __syncthreads();
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        C[i * N + j] = s_a[threadIdx.y] * s_b[threadIdx.x];
    }
}

extern "C" void solve(const float* a, const float* b, float* C, int M, int N) {
    dim3 threads(TILE_X, TILE_Y);
    dim3 blocks((N + TILE_X - 1) / TILE_X, (M + TILE_Y - 1) / TILE_Y);

    outer_product_kernel<<<blocks, threads>>>(a, b, C, M, N);
    cudaDeviceSynchronize();
}