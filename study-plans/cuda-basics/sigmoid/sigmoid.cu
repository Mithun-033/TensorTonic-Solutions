#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Numerically stable, branchless device function
__device__ __forceinline__ float safe_sigmoid(float x) {
    float z = __expf(-fabsf(x));
    float inv_denom = 1.0f / (1.0f + z);
    float num = (x >= 0.0f) ? 1.0f : z;
    
    return num * inv_denom;
}

__global__ void sigmoid_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        output[idx] = safe_sigmoid(input[idx]);
    }
}
extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}