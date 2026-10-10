def get_lr(it, max_lr, warmup_iters, max_iters):
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)

    if it < max_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)

    return 0.0