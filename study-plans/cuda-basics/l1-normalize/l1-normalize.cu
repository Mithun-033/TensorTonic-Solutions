#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_abs_sum(const float* input, float* sumv, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float v = 0.0f;
    if (i < N) {
        v = fabsf(input[i]);
    }
    s[tid] = v;
    __syncthreads();

    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (tid < st) {
            s[tid] += s[tid + st];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sumv, s[0]);
    }
}

__global__ void divide_by_scalar(const float* input, float* output, const float* sumv, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input[i] / (*sumv);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    reduce_abs_sum<<<blocks, threads>>>(input, d_sum, N);
    divide_by_scalar<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}