import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Perform one RMSProp update step.
    """
    w,g,s = np.asarray(w), np.asarray(g), np.asarray(s)
    s = beta * s + (1 - beta) * (g * g)
    w = w - (lr * g) / (np.sqrt(s) + eps)
    return w,s