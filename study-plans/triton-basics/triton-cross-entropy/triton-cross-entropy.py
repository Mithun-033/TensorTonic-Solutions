import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, loss_out_ptr,
    stride_logits_row,
    B, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis = 0)
    
    cols = tl.arange(0,BLOCK_SIZE)
    logits_ptrs = logits_ptr + pid * stride_logits_row + cols
    mask = cols < C

    logits = tl.load(logits_ptrs, mask = mask, other = float("-inf"))
    _max = tl.max(logits, axis = 0)
    shifted_logits = logits - _max
    log_sum_exp = _max + tl.log(tl.sum(tl.exp(shifted_logits), axis=0))

    target_id = tl.load(target_ptr + pid)
    target_logit = tl.load(logits_ptr + pid * stride_logits_row + target_id)

    loss = log_sum_exp - target_logit
    tl.atomic_add(loss_out_ptr, loss)
    


def solve(logits: torch.Tensor, target: torch.Tensor, loss_out: torch.Tensor) -> None:
    """Launch the cross-entropy kernel: one program per row, atomic accumulate, then divide by B."""
    B, C = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(C)
    loss_out.zero_()
    grid = (B,)
    cross_entropy_kernel[grid](
        logits, target, loss_out,
        logits.stride(0),
        B, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    loss_out.div_(B)