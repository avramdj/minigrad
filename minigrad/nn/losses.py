def cross_entropy(predictions, targets):
    """
    Computes cross entropy between targets and predictions.
    """
    n = predictions.shape[0]
    loss = -(targets * (predictions + 1e-9).log()).sum() / n
    return loss
