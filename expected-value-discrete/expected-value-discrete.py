import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    x = np.asarray(x, dtype = np.float32)
    p = np.asarray(p, dtype = np.float32)

    return np.sum(x * p, axis = 0, keepdims = False)