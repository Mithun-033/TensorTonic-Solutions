import torch
import triton
import triton.language as tl


@triton.jit
def fused_matmul_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)

    offset_m = BLOCK_M * pid_m + tl.arange(0,BLOCK_M)
    offset_n = BLOCK_N * pid_n + tl.arange(0,BLOCK_N)
    offset_k = tl.arange(0,BLOCK_K)

    a_ptrs = a_ptr + offset_m[:,None] * stride_am + offset_k[None,:] * stride_ak
    b_ptrs = b_ptr + offset_k[:,None] * stride_bk + offset_n[None,:] * stride_bn

    accum = tl.zeros((BLOCK_M,BLOCK_N), dtype = tl.float32)

    for k in range(0,K,BLOCK_K):
        mask_k = (k + offset_k) < K

        a = tl.load(a_ptrs, mask = (offset_m[:, None] < M) & (mask_k[None, :]), other = 0.0)
        b = tl.load(b_ptrs, mask = ((mask_k[:,None]) & (offset_n[None,:] < N)), other = 0.0)

        accum = tl.dot(a,b,accum, allow_tf32 = False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # bias
    bias_tile = tl.load(bias_ptr + offset_n, mask = offset_n < N, other = 0.0)
    accum += bias_tile[None,:]
    # ReLU
    accum = tl.maximum(accum,0.0)
    c_ptrs = c_ptr + offset_m[:,None] * stride_cm + offset_n[None,:] * stride_cn
    c_mask = (offset_m[:,None] < M) & (offset_n[None,:] < N)

    tl.store(c_ptrs, accum, mask = c_mask)


def solve(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> None:
    """Launch fused kernel: out = relu(A @ B + bias)."""
    M, K = A.shape
    K2, N = B.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    fused_matmul_bias_relu_kernel[grid](
        A, B, bias, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )