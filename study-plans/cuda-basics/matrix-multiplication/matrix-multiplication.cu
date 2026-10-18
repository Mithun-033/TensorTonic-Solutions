#include <cuda_runtime.h>
#define BM 128
#define BN 128
#define BK 32
#define TM 8
#define TN 8

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K){
    __shared__ float s_a[BM][BK] ;
    __shared__ float s_b[BK][BN] ;
    float reg_c[TM][TN] = {0.0f};
    
    const int c_row = blockIdx.y * BM + threadIdx.y * TM;
    const int c_col = blockIdx.x * BN + threadIdx.x * TN;
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; 
    
    constexpr int elements_per_thread_a = (BM * BK) / ((BN/TN) * (BM/TM));
    const int a_row = tid / (BK / elements_per_thread_a);
    const int a_col = elements_per_thread_a * (tid % (BK / elements_per_thread_a));

    constexpr int elements_per_thread_b = (BK * BN) / ((BM/TM) * (BN/TN));
    const int b_row = tid / (BN / elements_per_thread_b);
    const int b_col = elements_per_thread_b * (tid % (BN / elements_per_thread_b));

    for (int k_offset = 0; k_offset < K; k_offset += BK){
        #pragma unroll
        for (int i=0; i < elements_per_thread_a; i++){
            int g_a_row = blockIdx.y * BM + a_row;
            int g_a_col = (a_col + i) + k_offset;
            if (g_a_row < M && g_a_col < K){
                    s_a[a_row][a_col + i] = A[g_a_row * K + g_a_col];
            }
            else{
                s_a[a_row][a_col + i] = 0.0f;
            }
        }
        #pragma unroll
        for (int i = 0; i < elements_per_thread_b; ++i) {
            int g_b_row = k_offset + b_row;
            int g_b_col = blockIdx.x * BN + b_col + i;
            if (g_b_row < K && g_b_col < N) {
                s_b[b_row][b_col + i] = B[g_b_row * N + g_b_col];
            } else {
                s_b[b_row][b_col + i] = 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for(int dot_k = 0; dot_k < BK; ++dot_k){
            float reg_a[TM];
            float reg_b[TN];

            #pragma unroll
            for(int i=0; i < TM; ++i){
                reg_a[i] = s_a[threadIdx.y * TM + i][dot_k];
            }

            #pragma unroll
            for(int j=0; j < TN; ++j){
                reg_b[j] = s_b[dot_k][threadIdx.x * TN + j]; 
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            if ((c_row + i) < M && (c_col + j) < N) {
                C[(c_row + i) * N + (c_col + j)] = reg_c[i][j];
            }
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K){
    dim3 threads(BN / TN, BM / TM);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_kernel<<<blocks,threads>>>(A,B,C,M,N,K);
    cudaDeviceSynchronize();
}