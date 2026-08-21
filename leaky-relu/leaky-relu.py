import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Apply Leaky ReLU elementwise and return a NumPy array.
    """
    # Write code here
    arr = np.asarray(x, dtype = float)
    mask = arr > 0
    mask2 = arr < 0

    return mask * arr + alpha * mask2 * arr