import numpy as np


def time_delay_embedding(x, dim, lag):
    embedded = []
    max_idx = dim * lag
    for i in range(x.size - max_idx):
        embedded.append(x[i : i + max_idx : lag])
    return np.stack(embedded)
