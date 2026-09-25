import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    
    anchor = np.asarray(anchor, dtype = np.float32)
    positive = np.asarray(positive, dtype = np.float32)
    negative = np.asarray(negative, dtype = np.float32)

    if anchor.ndim == 1: anchor = anchor.reshape(1,-1)
    if positive.ndim == 1: positive = positive.reshape(1,-1)
    if negative.ndim == 1: negative = negative.reshape(1,-1)
    pos_dist = np.sum((anchor - positive) ** 2, axis = 1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)

    losses = np.maximum(0.0, pos_dist - neg_dist + margin)
    return float(np.mean(losses))