from typing import Iterable

from minigrad import Tensor


class Optimizer:
    def __init__(self, params: Iterable[Tensor]):
        self.params = params

    def step(self):
        raise NotImplementedError("Subclass didn't implement this function")

    def __call__(self, zero_grad=True):
        self.step()
        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
