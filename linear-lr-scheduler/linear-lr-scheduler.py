def linear_lr(step: int, total_steps: int, initial_lr: float, final_lr: float = 0.0, warmup_steps: int = 0) -> float:
    """
    Returns the learning rate as a float.
    """
    if step < warmup_steps:
        return initial_lr * (step / warmup_steps)
    elif step < total_steps:
        return initial_lr + ((step - warmup_steps) / (total_steps - warmup_steps)) * (final_lr - initial_lr)

    else:
        return final_lr
