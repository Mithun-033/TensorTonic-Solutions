import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    K = max(y) + 1 if num_classes is None else num_classes
    arr = np.zeros((len(y), K), dtype = np.float32)
    arr[np.arange(len(y)), y] = 1.0

    return arr