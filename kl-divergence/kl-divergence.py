import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    p = np.asarray(p, dtype = np.float32)
    q = np.asarray(q, dtype = np.float32)

    positive = p > 0
    q_safe = np.clip(q[positive], eps, None)
    return float(np.sum(p[positive] * np.log(p[positive] / q_safe)))