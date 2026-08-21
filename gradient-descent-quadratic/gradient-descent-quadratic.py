def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Return final x after 'steps' iterations.
    """
    for _ in range(steps):
        x0 = x0 - lr * (2*a*x0 + b)

    return x0