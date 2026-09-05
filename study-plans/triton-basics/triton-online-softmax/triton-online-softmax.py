import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)

    row_start = pid * N
    m = float("-inf")
    l = 0.0

    for block in range(0,N,BLOCK_SIZE):
        cols = block + tl.arange(0,BLOCK_SIZE)
        mask = cols < N
        chunk = tl.load(x_ptr + row_start + cols, mask = mask, other = float("-inf"))
        m_chunk = tl.max(chunk, axis = 0)
        m_new = tl.maximum(m,m_chunk)
        l = l*tl.exp(m-m_new) + tl.sum(tl.exp(chunk - m_new), axis = 0)
        m = m_new
    
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        chunk = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
        y = tl.exp(chunk - m) / l
        tl.store(out_ptr + row_start + cols, y, mask=mask)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the online softmax kernel row-wise."""
    M, N = x.shape
    BLOCK_SIZE = 1024
    grid = (M,)
    online_softmax_kernel[grid](x, out, M, N, BLOCK_SIZE=BLOCK_SIZE)