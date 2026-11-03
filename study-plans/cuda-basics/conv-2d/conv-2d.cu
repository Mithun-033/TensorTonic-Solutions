#include <cuda_runtime.h>

#define TILE_H 16
#define TILE_W 16
#define K 9

__constant__ float c_kernel[K * K];

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int H, int W, int kH, int kW)
{
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    __shared__ float s_tile[TILE_H + K - 1][TILE_W + K - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_i = blockIdx.y * TILE_H + ty;
    int out_j = blockIdx.x * TILE_W + tx;

    int input_tile_h = TILE_H + kH - 1;
    int input_tile_w = TILE_W + kW - 1;
    int total_load = input_tile_h * input_tile_w;

    int threads_per_block = blockDim.x * blockDim.y;
    int tid = ty * blockDim.x + tx;

    int block_x_ptr = blockIdx.x * TILE_W;
    int block_y_ptr = blockIdx.y * TILE_H;

    for (int id = tid; id < total_load; id += threads_per_block) {
        int smem_y = id / input_tile_w;
        int smem_x = id % input_tile_w;
        
        int glob_y = block_y_ptr + smem_y;
        int glob_x = block_x_ptr + smem_x;

        if (glob_y < H && glob_x < W) {
            s_tile[smem_y][smem_x] = input[glob_y * W + glob_x];
        } else {
            s_tile[smem_y][smem_x] = 0.0f;
        }
    }
    __syncthreads();

    if (out_i < outH && out_j < outW) {
        float acc = 0.0f;
        #pragma unroll
        for (int x = 0; x < kH; ++x) {
            #pragma unroll
            for (int y = 0; y < kW; ++y) {
                acc += s_tile[ty + x][tx + y] * c_kernel[x * kW + y];
            }
        }
        output[out_i * outW + out_j] = acc;
    }
}

extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int H, int W, int kH, int kW)
{
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    cudaMemcpyToSymbol(c_kernel, kernel, kH * kW * sizeof(float));

    dim3 threads(TILE_W, TILE_H);
    dim3 blocks((outW + TILE_W - 1) / TILE_W, (outH + TILE_H - 1) / TILE_H);

    conv2d_kernel<<<blocks, threads>>>(input, output, H, W, kH, kW);
    cudaDeviceSynchronize();
}