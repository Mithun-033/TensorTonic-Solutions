import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, gamma_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)
    cols = tl.arange(0,BLOCK_SIZE)
    mask = cols < N

    x_ptrs = x_ptr + pid * stride_x_row + cols
    out_ptrs = out_ptr + pid * stride_out_row + cols
    gamma = tl.load(gamma_ptr + cols, mask = mask, other = 0.0)

    x = tl.load(x_ptrs, mask = mask, other = 0.0)
    div = 1 / tl.sqrt((tl.sum(x * x) / N) + eps)
    out = div * x * gamma

    tl.store(out_ptrs, out, mask = mask)

def solve(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the RMSNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_fwd_kernel[grid](
        x, gamma, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )