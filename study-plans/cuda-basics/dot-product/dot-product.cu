#include <cuda_runtime.h>

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float v = 0.0f;
    if (i < N) {
        v = A[i] * B[i];
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
        atomicAdd(result, s[0]);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    dot_kernel<<<blocks, threads>>>(A, B, result, N);
    cudaDeviceSynchronize();
}