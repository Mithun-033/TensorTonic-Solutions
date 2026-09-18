import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    x = np.asarray(x, dtype = np.float32)
    gamma = np.asarray(gamma, dtype = np.float32)
    beta = np.asarray(beta, dtype = np.float32)

    if x.ndim == 2:
        axis = (0,)
        param_shape = (1,-1)

    else:
        axis = (0,2,3)
        param_shape = (1,-1,1,1)

    mean = np.mean(x, axis = axis, keepdims = True)
    variance = np.var(x, axis = axis, keepdims = True)
    normalized = (x - mean) / (np.sqrt(variance + eps))

    return normalized * gamma.reshape(param_shape) + beta.reshape(param_shape)