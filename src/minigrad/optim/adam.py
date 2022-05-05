from typing import Iterable
from minigrad import Tensor
from .optimizer import Optimizer
import minigrad


class Adam(Optimizer):
    """
    Adam implementation.
    The Adaptive Moment Estimation algorithm for stochastic gradient descent.
    see: https://arxiv.org/abs/1412.6980
    """
    def __init__(self, params: Iterable[Tensor], learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        # First moment vector
        self.m0 = [Tensor.zeros(p.shape, device=p.device, requires_grad=False) for p in params]
        # Second moment vector
        self.v0 = [Tensor.zeros(p.shape, device=p.device, requires_grad=False) for p in params]
        # Time step
        self.t = 0

    def step(self):
        self.t += 1
        g = self.learning_rate * ((1 - self.b2 ** self.t) ** 0.5) / (1 - self.b1 ** self.t)
        with minigrad.no_grad():
            for i, param in enumerate(self.params):
                self.m0[i] = self.b1 * self.m0[i] + (1.0 - self.b1) * param.grad
                self.v0[i] = self.b2 * self.v0[i] + (1.0 - self.b2) * param.grad * param.grad
                param -= g * self.m0[i] / (self.v0[i].sqrt() + self.eps)
