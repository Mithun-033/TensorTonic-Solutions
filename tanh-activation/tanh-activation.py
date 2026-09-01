import numpy as np

def tanh(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype = np.float32)
    a = np.exp(x)
    b = np.exp(-x)

    return (a-b) / (a+b)