from minigrad.nn.module import Module

from minigrad.nn.transform import one_hot_encode


class MSE(Module):
    def __init__(self, n_classes=None):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x, y):
        return mse(x, y, C=self.n_classes)


class CrossEntropyLoss(Module):
    def __init__(self, n_classes=None):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x, y):
        return cross_entropy(x, y, C=self.n_classes)


class BinaryCrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return binary_cross_entropy(x, y)


def mse(targets, predictions, one_hot=False, C=None):
    if not one_hot:
        targets = one_hot_encode(targets, C=C)
    return ((targets - predictions) ** 2).mean()


def cross_entropy(targets, predictions, eps=1e-12, one_hot=False, C=None):
    """
    Calculates categorical cross entropy between targets and predictions.
    """
    # TODO: fix
    if not one_hot:
        targets = one_hot_encode(targets, C=C)
    predictions = predictions.clip(eps, 1 - eps)
    loss = (targets * predictions.log()).sum() / predictions.shape[1]
    return -loss


def binary_cross_entropy(targets, predictions, eps=1e-12):
    """
    Calculates binary cross entropy between targets and predictions.
    """
    predictions = predictions.clip(eps, 1 - eps)
    n = predictions.shape[1]
    p_log = predictions.log()
    loss = -(targets * p_log + (1 - targets) * (1 - p_log)) / n
    return loss.sum()
