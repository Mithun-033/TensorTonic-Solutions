#include <cuda_runtime.h>

__global__ void sum_and_sumsq_kernel(const float* input, float* sum_out, float* sumsq_out, int N) {
    __shared__ float s_sum[256];
    __shared__ float s_sq[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * 256 + tid;

    float v = (i < N) ? input[i] : 0.0f;
    s_sum[tid] = v;
    s_sq[tid] = v * v;
    __syncthreads();

    #pragma unroll
    for (int st = 256 >> 1; st > 0; st >>= 1) {
        if (tid < st) {
            s_sum[tid] += s_sum[tid + st];
            s_sq[tid] += s_sq[tid + st];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_out, s_sum[0]);
        atomicAdd(sumsq_out, s_sq[0]);
    }
}

__global__ void finalize_kernel(const float* sum, const float* sumsq, float* mean_out, float* var_out, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float invN = 1.0f / (float)N;
        float mu = (*sum) * invN;
        float ex2 = (*sumsq) * invN;
        float var = ex2 - mu * mu;
        if (var < 0.0f) var = 0.0f;
        *mean_out = mu;
        *var_out = var;
    }
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    float *d_sum, *d_sumsq;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_sumsq, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    cudaMemset(d_sumsq, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sum_and_sumsq_kernel<<<blocks, threads>>>(input, d_sum, d_sumsq, N);
    finalize_kernel<<<1, 1>>>(d_sum, d_sumsq, mean_out, var_out, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
    cudaFree(d_sumsq);
}