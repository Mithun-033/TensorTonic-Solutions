#include <cuda_runtime.h>
#define TILE_DIM 32
#define block_rows 8

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for(int i = 0; i < TILE_DIM; i += block_rows){
        if ((y + i) < M && x < N){
            tile[threadIdx.y + i][threadIdx.x] = A[(y + i) * N + x];
        }
    }

    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    #pragma unroll
    for(int r=0; r < TILE_DIM; r += block_rows){
        if ((y+r) < N && x < M) {
            B[(y+r) * M + x] = tile[threadIdx.x][threadIdx.y + r];
        }
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(TILE_DIM, block_rows);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
