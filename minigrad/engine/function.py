from __future__ import annotations

from os import getenv
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from minigrad.engine.buffer import Buffer

if TYPE_CHECKING:
    from minigrad.engine.tensor import Tensor


class Function:
    def __init__(self) -> None:
        self.saved_tensors: List[Tensor] = []

    def forward(self, *args: Any) -> Buffer:
        raise NotImplementedError("forward method must be implemented by all subclasses of Function")

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        raise NotImplementedError("backward method must be implemented by all subclasses of Function")

    def __call__(self, *args: Tensor) -> Buffer:
        # ensire all tensors on the same device
        for arg in args:
            if arg.device != args[0].device:
                raise ValueError("All tensors must be on the same device")
        data = self.forward(*args)
        requires_grad = any(arg.requires_grad for arg in args)
        if requires_grad:
            self.saved_tensors = list(args)
        return data


class Add(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a, b = args
        data = a.data + b.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return grad_output, grad_output


class Sub(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a, b = args
        data = a.data - b.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return grad_output, -grad_output


class Mul(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a, b = args
        self.a = a
        self.b = b
        data = a.data * b.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return grad_output * self.b, grad_output * self.a


class Div(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a, b = args
        self.a = a
        self.b = b
        data = a.data / b.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return grad_output / self.b, -grad_output * self.a / (self.b**2)


class Pow(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a, exponent = args
        self.a = a
        self.exponent = exponent.data
        data = a.data**exponent.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return self.exponent * (self.a.data ** (self.exponent - 1)) * grad_output, None


class Exp(Function):
    def forward(self, *args: Tensor) -> Buffer:
        self.a = args[0]
        data = self.a.data.exp()
        requires_grad = self.a.requires_grad
        # TODO: check if this is necessary, seems bloated
        if not getenv("MEMORY_OPTIMIZED", False):
            self.save_for_backward = data
        return Tensor(data, requires_grad=requires_grad, device=self.a.device)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if self.save_for_backward is not None:
            data = self.save_for_backward
        else:
            data = self.a.data.exp()
        return (data * grad_output,)


class MatMul(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a, b = args
        self.a = a
        self.b = b
        data = a.data @ b.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return grad_output @ self.b.T(), self.a.T() @ grad_output


class Neg(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a = args[0]
        data = -a.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (-grad_output,)


class Mean(Function):
    def __init__(self, axis=0) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, *args: Tensor) -> Buffer:
        a = args[0]
        self.a = a
        data = a.data.mean(axis=self.axis)
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (grad_output / self.a.data.size(),)
