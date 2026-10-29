#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int kN) {
    int i = blockIdx.x * 256 + threadIdx.x;
    int outN = N - kN + 1;
    if(i < outN){
        float acc = 0.0f;
        for(int j = 0; j < kN; j++){
            acc += input[i+j] * kernel[j];
        }
        output[i] = acc;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int threads = 256;
    dim3 blocks((outN + 255) / 256);
    conv1d_kernel<<<blocks, threads>>>(input, kernel, output, N, kN);
    cudaDeviceSynchronize();
}
