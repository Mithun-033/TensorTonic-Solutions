import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    stride_x_row, stride_out_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)
    cols = tl.arange(0,BLOCK_SIZE)
    mask = cols < N

    x_ptrs = x_ptr + pid * stride_x_row + cols
    out_ptrs = out_ptr + pid * stride_out_row + cols

    x = tl.load(x_ptrs, mask = mask, other = 0.0)
    
    sum_x = tl.sum(x, axis = 0)
    mean = sum_x / N
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis = 0) / N
    

    gamma = tl.load(gamma_ptr + cols, mask = mask, other = 0.0)
    beta = tl.load(beta_ptr + cols, mask = mask, other = 0.0)

    out = (centered / (tl.sqrt(var + eps))) * gamma + beta
    tl.store(out_ptrs, out, mask = mask)
    

def solve(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, out: torch.Tensor, eps: float) -> None:
    """Launch the LayerNorm forward kernel: one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    layernorm_fwd_kernel[grid](
        x, gamma, beta, out,
        x.stride(0), out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )