import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    matrix = np.asarray(matrix, dtype = np.float32)
    if axis is None:
            axis = (0,1)
    if norm_type == "l1":
        norms = np.sum(np.abs(matrix), axis = axis, keepdims = True)

    elif norm_type == "l2":
        norms = np.sqrt(np.sum(matrix ** 2, axis = axis, keepdims = True))

    else:
        norms = np.max(np.abs(matrix), axis = axis, keepdims = True)

    norms = np.where(norms == 0, 1.0, norms)

    return matrix / norms
        
            