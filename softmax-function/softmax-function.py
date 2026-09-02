import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        shifted = x - np.max(x)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(x)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)