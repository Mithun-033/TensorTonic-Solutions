import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Return a causally masked copy of the attention scores.
    """
    scores = np.asarray(scores, dtype=float)
    sequence_length = scores.shape[-1]
    mask = np.tril(np.ones((sequence_length,sequence_length), dtype=bool))
    return np.where(mask, scores, mask_value)
    