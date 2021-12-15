from typing import Iterable
from minigrad import Tensor
import minigrad

class SDG:
    def __init__(self, params: Iterable[Tensor], learning_rate=1e-3):
        self.params = params
        self.learning_rate = learning_rate

    def step(self):
        with minigrad.no_grad():
            for param in self.params:
                param.data -= param.grad.data * self.learning_rate
        for param in self.params:
            param.zero_grad()