import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    arr = np.asarray(x, dtype = float)
    mask = arr > 0
    return arr * mask