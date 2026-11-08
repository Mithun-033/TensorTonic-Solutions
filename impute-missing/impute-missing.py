import numpy as np

def _impute_feature(feature, strategy):
    missing = np.isnan(feature)
    observed = feature[~missing]
    fill = 0.0 if observed.size == 0 else float(np.mean(observed) if strategy == "mean" else np.median(observed))
    feature[missing] = fill
    return feature
    
def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    result = np.asarray(X, dtype=float).copy()
    if result.ndim == 1:
        return _impute_feature(result, strategy)
    for col in range(result.shape[1]):
        _impute_feature(result[:, col], strategy)
    return result