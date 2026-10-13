#include <cuda_runtime.h>

// Vectorized kernel for 16-byte chunks
__global__ void vector_add_float4(const float* A, const float* B, float* C, int N_vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_vec) {
        const float4* A_vec = reinterpret_cast<const float4*>(A);
        const float4* B_vec = reinterpret_cast<const float4*>(B);
        float4* C_vec = reinterpret_cast<float4*>(C);

        float4 a = A_vec[i];
        float4 b = B_vec[i];
        
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        C_vec[i] = c;
    }
}
// Scalar fallback kernel for remaining elements (N % 4)
__global__ void vector_add_tail(const float* A, const float* B, float* C, int start_idx, int N) {
    int i = start_idx + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threads = 256;
    int N_vec = N / 4; 

    if (N_vec > 0) {
        int blocks_vec = (N_vec + threads - 1) / threads;
        vector_add_float4<<<blocks_vec, threads>>>(A, B, C, N_vec);
    }
    int remainder = N % 4;
    if (remainder > 0) {
        int start_idx = N_vec * 4;
        vector_add_tail<<<1, remainder>>>(A, B, C, start_idx, N);
    }

    cudaDeviceSynchronize();
}