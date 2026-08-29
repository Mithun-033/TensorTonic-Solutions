import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    dk = Q.shape[-1]

    logits = (Q @ K.transpose(-2,-1) ) / math.sqrt(dk)
    attn_logits = F.softmax(logits, dim = -1)
    output = attn_logits @ V
    return output
    