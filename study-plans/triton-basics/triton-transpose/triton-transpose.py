import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    a_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)

    offset_m = BLOCK_M * pid_m + tl.arange(0,BLOCK_M)
    offset_n = BLOCK_N * pid_n + tl.arange(0,BLOCK_N)

    a_ptrs = a_ptr + offset_m[:,None] * stride_am + offset_n[None,:] * stride_an
    a_mask = (offset_m[:,None] < M) and (offset_n[None,:] < N)
    a = tl.load(a_ptrs, mask = a_mask, other = 0.0)

    out_ptrs = out_ptr + offset_n[None,:] * stride_om + offset_m[:,None] * stride_on
    tl.store(out_ptrs, a, mask = a_mask)
    
def solve(A: torch.Tensor, out: torch.Tensor) -> None:
    """Launch transpose_kernel: out[j, i] = A[i, j]."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    transpose_kernel[grid](
        A, out,
        M, N,
        A.stride(0), A.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )