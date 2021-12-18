from minigrad import nn
from .transform import one_hot_encode


class CrossEntropy(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x, y):
        return cross_entropy(x, y, C=self.n_classes)


def cross_entropy(predictions, targets, eps=1e-12, one_hot=False, C=None):
    """
    Computes cross entropy between targets and predictions.
    """
    if not one_hot:
        targets = one_hot_encode(targets, C=C)
    predictions = predictions.clip(eps, 1-eps)
    n = predictions.shape[0]
    loss = -(targets * predictions.log()).sum(axis=1) / n
    return loss.sum(axis=None)
