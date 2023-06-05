from numbers import Number
from typing import ClassVar, Union

import numpy as np

from minigrad.engine.buffer import create_buffer

Tensorable = Union["Tensor", Number, np.ndarray, list]


# pytorch-like tensor class
class Tensor:
    device: ClassVar[str] = "cpu"

    def __init__(self, data: Tensorable, requires_grad=False, device=device):
        self.data = create_buffer(data, device)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    # fmt: off
    def __repr__(self) -> str: return f"Tensor({self.data})"
    def __add__(self, other: Tensorable): return self._add(other)
    def __radd__(self, other: Tensorable): return self._add(other)
    def __sub__(self, other: Tensorable): return self._sub(other)
    def __rsub__(self, other: Tensorable): return self._sub(other)
    def __mul__(self, other: Tensorable): return self._mul(other)
    def __rmul__(self, other: Tensorable): return self._mul(other)
    def __truediv__(self, other: Tensorable): return self._div(other)
    def __rtruediv__(self, other: Tensorable): return self._div(other)
    def __pow__(self, other: Tensorable): return self._pow(other)
    def __rpow__(self, other: Tensorable): return self._pow(other)
    def __matmul__(self, other: Tensorable): return self._matmul(other)
    def __rmatmul__(self, other: Tensorable): return self._matmul(other)
    def __neg__(self): return self._neg()
    def __getitem__(self, index: slice): return self.data[index]
    # fmt: on
