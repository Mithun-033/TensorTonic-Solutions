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
    int tid = blockIdx.x * 256 + threadIdx.x;
    if(tid  < N){
    output[tid] = rsqrtf(*sumv) * input[tid];
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
