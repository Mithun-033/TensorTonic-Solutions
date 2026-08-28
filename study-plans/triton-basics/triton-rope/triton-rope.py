import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    N, D,
    BLOCK_SIZE: tl.constexpr,
):
    rows = tl.program_id(axis = 0)
    cols = tl.arange(0,BLOCK_SIZE)

    half = D // 2
    mask = cols < half

    x_even = tl.load(x_ptr + rows * D + 2 * cols, mask = mask, other = 0.0)
    x_odd = tl.load(x_ptr + rows * D + 2 * cols + 1, mask = mask, other = 0.0)
    c = tl.load(cos_ptr + rows * half + cols, mask=mask, other=0.0)
    s = tl.load(sin_ptr + rows * half + cols, mask=mask, other=0.0)

    out_even = x_even * c - x_odd * s
    out_odd = x_even * s + x_odd * c

    tl.store(out_ptr + rows * D + 2 * cols, out_even, mask=mask)
    tl.store(out_ptr + rows * D + 2 * cols + 1, out_odd, mask=mask)

def solve(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the RoPE kernel: rotate pairs of channels with per-row (cos, sin) tables."""
    N, D = x.shape
    BLOCK_SIZE = 1
    while BLOCK_SIZE < D // 2:
        BLOCK_SIZE *= 2
    if BLOCK_SIZE < 1:
        BLOCK_SIZE = 1
    grid = (N,)
    rope_kernel[grid](x, cos, sin, out, N, D, BLOCK_SIZE=BLOCK_SIZE)