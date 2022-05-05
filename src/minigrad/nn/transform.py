import numpy as np


def one_hot_encode(targets, C=None):
    """
    Returns a tensor of shape (N,C) where C = num classes
    for a given tensor of shape (N,)
    """
    Cmax = int(targets.max().data.item() + 1)
    N = targets.shape[0]
    C = max(Cmax, C) if C else Cmax
    ret = targets.zeros((N, C), requires_grad=False)
    idx = np.array(targets.data, dtype=np.int)
    ret.data[np.arange(N), idx] = 1
    return ret
