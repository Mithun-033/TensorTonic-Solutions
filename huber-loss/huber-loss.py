import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    y_true = np.asarray(y_true, dtype = np.float32)
    y_pred = np.asarray(y_pred, dtype = np.float32)

    error = y_true - y_pred
    L = np.where(abs(error) > delta, delta * (abs(error) - delta / 2), (error ** 2) / 2)

    return L.mean().item()