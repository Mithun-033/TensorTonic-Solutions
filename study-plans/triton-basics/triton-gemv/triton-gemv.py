import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis = 0)
    
    offset_m = BLOCK_M * pid + tl.arange(0,BLOCK_M)
    offset_n = tl.arange(0,BLOCK_N)
    
    mask_m = offset_m < M
    accum = tl.zeros((BLOCK_M,), dtype = tl.float32)

    for n in range(0,N,BLOCK_N):
        offs_n = n + offset_n
        n_mask = offs_n < N

        a_ptrs = a_ptr + offset_m[:,None] * stride_am + offs_n[None,:] * stride_an
        tile = tl.load(a_ptrs, mask = mask_m[:,None] * n_mask[None,:], other = 0.0)

        vector_tile = tl.load(x_ptr + offs_n, mask = n_mask, other = 0.0)
        accum += tl.sum(tile * vector_tile[None,:], axis = 1)

    tl.store(out_ptr + offset_m, accum, mask = mask_m)
    


def solve(A: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gemv_kernel: out = A @ x."""
    M, N = A.shape
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    gemv_kernel[grid](
        A, x, out,
        M, N,
        A.stride(0), A.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )