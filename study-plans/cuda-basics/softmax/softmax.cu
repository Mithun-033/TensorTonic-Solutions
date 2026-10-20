#include <cuda_runtime.h>
#include <float.h>
#define REDUCE_BLOCK_SIZE 256

__global__ void reduce_max(const float* in, float* out, int N){
    __shared__ float s[REDUCE_BLOCK_SIZE];
    int tid = threadIdx.x;
    float v = -FLT_MAX;
    for(int i=tid; i < N; i += REDUCE_BLOCK_SIZE){
        v = fmaxf(v, in[i]);
    }
    s[tid] = v;
    __syncthreads();
    for(int st = REDUCE_BLOCK_SIZE >> 1; st > 0; st >>= 1){
        if (tid < st){
            s[tid] = fmaxf(s[tid], s[tid + st]);
        }
        __syncthreads();
    }
    if(tid == 0) *out = s[0];
}
__global__ void reduce_sum(const float* in, float* out, const float* d_max, float* d_sum, int N){
    __shared__ float s[REDUCE_BLOCK_SIZE];
    int tid = threadIdx.x;
    
    float local = 0.0f;
    float m = *d_max;
    
    for(int i = tid; i < N; i += REDUCE_BLOCK_SIZE){
        float ans = expf(in[i] - m);
        out[i] = ans;
        local += ans;
    }
    s[tid] = local;
    __syncthreads();

    for(int i = REDUCE_BLOCK_SIZE >> 1; i > 0; i >>= 1){
        if(tid < i){
            s[tid] = s[tid] + s[tid + i];
        }
        __syncthreads();
    }
    if(tid == 0) *d_sum = s[0];
}
__global__ void normalize(float* out, const float* sumv, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] /= *sumv;
}


extern "C" void solve(const float* input, float* output, int N) {
    float *d_max, *d_sum;
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    reduce_max<<<1,REDUCE_BLOCK_SIZE>>>(input, d_max, N);
    reduce_sum<<<1,REDUCE_BLOCK_SIZE>>>(input,output,d_max,d_sum,N);

    int threads = 256;
    int blocks = ((N + threads - 1) / threads);
    normalize<<<blocks,threads>>>(output, d_sum, N);
    
    cudaDeviceSynchronize();
    cudaFree(d_max);
    cudaFree(d_sum);
}