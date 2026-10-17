#include <cuda_runtime.h>
#define TILE_DIM 32

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (y < M && x < N){
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }

    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (y < N && x < M) {
        B[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
