#include <cuda_runtime.h>
#include <math.h>

__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {
    __shared__ float sq_sum[256];
    int tid = threadIdx.x;
    float local = 0.0f;
    
    for(int i = tid;i < N; i += 256){
        float v = input[i];
        local += v * v;
    }
    sq_sum[tid] = local;
    __syncthreads();

    for(int stride = 256 >> 1; stride > 0; stride >>= 1){
        if(tid < stride){
            sq_sum[tid] += sq_sum[stride + tid];
        }
        __syncthreads();
    }

    if(tid == 0){
        *sumv = sq_sum[0];
    }
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;

    if (idx >= N) return;
    float inv_sqrt = rsqrtf(*sumv);

    if (idx + 3 < N) {
        float4 v = reinterpret_cast<const float4*>(input)[tid];
        v.x *= inv_sqrt;
        v.y *= inv_sqrt;
        v.z *= inv_sqrt;
        v.w *= inv_sqrt;
        reinterpret_cast<float4*>(output)[tid] = v;
    } else {
        for (int i = 0; idx + i < N; ++i) {
            output[idx + i] = input[idx + i] * inv_sqrt;
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    reduce_sq_sum<<<1, 256>>>(input, d_sum, N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}
