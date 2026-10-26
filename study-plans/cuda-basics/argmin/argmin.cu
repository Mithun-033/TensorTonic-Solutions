#include <cuda_runtime.h>
#include <float.h>

__global__ void argmin_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    __shared__ float s_val[256];
    __shared__ float s_idx[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * 256 + tid;

    float val = FLT_MAX;
    int idx = INT_MAX;
    if (i < N){
        val = input[i];
        idx = i;
    }
    s_val[tid] = val;
    s_idx[tid] = idx;
    __syncthreads();

    for(int stride = 128; stride > 0; stride >>= 1){
        if(tid < stride){
            float va = s_val[tid];
            float vb = s_val[tid + stride];
            int idxa = s_idx[tid];
            int idxb = s_idx[tid + stride];

            if ((vb < va) || (va == vb && idxb < idxa)){
                s_val[tid] = vb;
                s_idx[tid] = idxb;
            }
        }
        __syncthreads();
    }
    if(tid == 0){
        block_vals[blockIdx.x] = s_val[0];
        block_idxs[blockIdx.x] = s_idx[0];
    }
}

__global__ void argmin_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float s_val[256];
    __shared__ float s_idx[256];

    int tid = threadIdx.x;
    float best_v = FLT_MAX;
    float best_idx = INT_MAX;
    
    for(int k = tid; k < num_blocks; k += 256){
        float v = block_vals[k];
        int idx = block_idxs[k];

        if(v < best_v || (v == best_v && idx < best_idx)){
            best_v = v;
            best_idx = idx;
        }
    }
    s_val[tid] = best_v;
    s_idx[tid] = best_idx;
    __syncthreads();

    for(int stride = 128; stride > 0; stride >>= 1){
        if(tid < stride){
            float va = s_val[tid];
            float vb = s_val[tid + stride];
            int idxa = s_idx[tid];
            int idxb = s_idx[tid + stride];

            if ((vb < va) || (va == vb && idxb < idxa)){
                s_val[tid] = vb;
                s_idx[tid] = idxb;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = s_idx[0];
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmin_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmin_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
