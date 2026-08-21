import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    arr = np.asarray(A)
    m,n = len(arr), len(arr[0])
    transpose = np.empty((n,m))

    for i in range(n):
        for j in range(m):
            transpose[i,j] = arr[j,i]

    return transpose