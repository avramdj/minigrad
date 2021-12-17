def cross_entropy(predictions, targets, eps=1e-12):
    """
    Computes cross entropy between targets and predictions.
    """
    predictions = predictions.clip(eps, 1-eps)
    n = predictions.shape[0]
    loss = -(targets * predictions.log()).sum() / n
    return loss
