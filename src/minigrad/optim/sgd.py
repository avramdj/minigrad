from typing import Iterable
from minigrad import Tensor
from .optimizer import Optimizer
import minigrad


class SGD(Optimizer):
    """
    Vanilla implementation of Stochastic Gradient Descent
    """
    def __init__(self, params: Iterable[Tensor], learning_rate=1e-3):
        super().__init__(params)
        self.learning_rate = learning_rate

    def step(self):
        with minigrad.no_grad():
            for param in self.params:
                param -= param.grad * self.learning_rate
