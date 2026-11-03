#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int outw = W - kW + 1;
    int outh = H - kH + 1;

    if(i < outh && j < outw){
        float acc = 0.0f;
        for(int x=0; x < kH ; x++){
            for(int y=0; y < kW; y++){
                acc += kernel[x * kW + y] * input[(i + x) * W + (y + j)];
            }
        }
        output[i * outw + j] = acc;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    conv2d_kernel<<<blocks, threads>>>(input, kernel, output, H, W, kH, kW);
    cudaDeviceSynchronize();
}
