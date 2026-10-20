#include <cuda_runtime.h>
#include <math.h>
#define BLOCK_SIZE 256
__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float sq_sum[BLOCK_SIZE];
    __shared__ float smean;
    __shared__ float s_inv_std;
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for(int i = tid; i < N; i += BLOCK_SIZE){
        float val = input[row * N + i];
        local_sum += val;
        local_sq_sum += val * val;
    }
    s_sum[tid] = local_sum;
    sq_sum[tid] = local_sq_sum;

    __syncthreads();

    for(int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1){
        if(tid < stride){
            s_sum[tid] += s_sum[tid + stride];
            sq_sum[tid] += sq_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        float mean = s_sum[0] / (float)N;
        float var = fmaxf(0.0f, sq_sum[0] / (float)N - mean * mean);
        smean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = smean;
    float inv_std = s_inv_std;

    for(int i = tid; i < N; i += BLOCK_SIZE){
        float v = input[row * N + i];
        output[row * N + i] = gamma[i] * (v - mean) * inv_std + beta[i];
    }
    
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = BLOCK_SIZE;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
