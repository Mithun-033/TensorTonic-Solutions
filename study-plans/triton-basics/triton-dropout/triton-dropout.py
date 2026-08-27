import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr, mask_ptr, out_ptr,
    n, p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)
    offset = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    overflow_mask = offset < n

    x = tl.load(x_ptr + offset, mask = overflow_mask, other = 0.0)
    mask = tl.load(mask_ptr + offset, mask = overflow_mask, other = 0.0)

    out = x * mask * (1/(1-p))
    tl.store(out_ptr + offset, out, mask = overflow_mask)

def solve(x: torch.Tensor, mask: torch.Tensor, out: torch.Tensor, p: float) -> None:
    """Launch the dropout kernel: 1D grid over the input vector."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    dropout_kernel[grid](
        x, mask, out,
        n, p,
        BLOCK_SIZE=BLOCK_SIZE,
    )