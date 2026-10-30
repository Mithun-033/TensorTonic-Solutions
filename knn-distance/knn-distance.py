import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    train = np.asarray(X_train, dtype = np.float32)
    test = np.asarray(X_test, dtype = np.float32)

    if train.ndim == 1:
        train = train[:,None]
    if test.ndim == 1:
        test = test[:,None]

    dist = np.sqrt(np.sum((train[None,:,:] - test[:,None,:]) ** 2, axis = 2,keepdims = False)) # (n_test, n_train)
    k_act = min(k,train.shape[0])
    knn = np.argsort(dist, axis = 1)[:,:k_act] #(n_test, k_act)

    if k_act < k:
        padding = np.zeros((test.shape[0], k - k_act), dtype = int)
        padding = np.where(padding == 0, -1, -1)
        knn = np.concatenate((knn, padding), axis = 1)

    return knn.astype(int)