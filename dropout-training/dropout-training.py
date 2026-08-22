import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x)
    if rng:
        logits = rng.random(x.shape)
    else:
        logits = np.random.random(x.shape)

    if p == 0:
        return x, np.ones(x.shape)
    mask = np.where(logits >= p, 1/(1-p), 0)
    return x * mask, mask