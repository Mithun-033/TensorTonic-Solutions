#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* __restrict__ input, const float* __restrict__ gamma, float* __restrict__ output, int M, int N, float eps) {
    __shared__ float s_sum[256];
    __shared__ float s_inv_rms;
    
    int tid = threadIdx.x;
    const float* start = input + blockIdx.x * N;
    float* out_start = output + blockIdx.x * N;
    float local_sum = 0.0f;

    float local_input[16];
    int counter = 0;
    
    #pragma unroll
    for(int i = tid; i < N; i += 256){
        float val = start[i];
        local_sum += val * val;
        local_input[counter++] = val;
    }
    
    counter = 0;
    s_sum[tid] = local_sum;
    __syncthreads();

    for(int stride = 256 >> 1; stride > 0; stride >>= 1){
        if(tid < stride){
            s_sum[tid] += s_sum[stride + tid];
        }
        __syncthreads();
    }

    if(tid == 0){
        float mean_sq = s_sum[0] / (float)N;
        s_inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    float inv = s_inv_rms;
    #pragma unroll
    for(int i = tid; i < N; i += 256){
        out_start[i] = local_input[counter++] * inv * gamma[i];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}