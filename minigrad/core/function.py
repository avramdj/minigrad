from __future__ import annotations

from os import getenv
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from minigrad.core.buffer import Buffer
from minigrad.core.context import is_grad

if TYPE_CHECKING:
    from minigrad.core.tensor import Tensor


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
        if is_grad() and any(arg.requires_grad for arg in args):
            self.ctx: List[Any] = list(args)
        data = self.forward(*args)
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
        data = a.data * b.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return grad_output * self.ctx[1], grad_output * self.ctx[0]


class Div(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a, b = args
        data = a.data / b.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return grad_output / self.ctx[1], -grad_output * self.ctx[0] / (self.ctx[1] ** 2)


class Pow(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a, exponent = args
        data = a.data**exponent.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return grad_output * self.ctx[1] * (self.ctx[0].data ** (self.ctx[1] - 1.0)), None


class Exp(Function):
    def forward(self, *args: Tensor) -> Buffer:
        self.a = args[0]
        data = self.a.data.exp()
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (self.ctx[0].data.exp() * grad_output,)


class MatMul(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a, b = args
        data = a.data @ b.data
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return grad_output @ self.ctx[1].T(), self.ctx[0].T() @ grad_output


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
        data = a.data.mean(axis=self.axis)
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (grad_output / self.ctx[0].data.size(),)


class Log(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a = args[0]
        data = a.data.log()
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (grad_output / self.ctx[0].data,)


class Sqrt(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a = args[0]
        data = a.data.sqrt()
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (1 / 2 * grad_output / self.ctx[0].data.sqrt(),)


class Tanh(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a = args[0]
        data = a.data.tanh()
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (grad_output * (1 - self.ctx[0].data.tanh() ** 2),)


class ReLU(Function):
    def forward(self, *args: Tensor) -> Buffer:
        a = args[0]
        data = a.data.relu()
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (grad_output * (self.ctx[0].data > 0),)

class Flatten(Function):
    def __init__(self, start_dim) -> None:
        super().__init__()
        self.start_dim = start_dim

    def forward(self, *args: Tensor) -> Buffer:
        a = args[0]
        data = a.data.flatten(self.start_dim)
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (grad_output.reshape(self.ctx[0].shape),)
    
class Reshape(Function):
    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, *args: Tensor) -> Buffer:
        a = args[0]
        data = a.data.reshape(self.shape)
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (grad_output.reshape(self.ctx[0].shape),)
    
class Sum(Function):
    def __init__(self, axis) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, *args: Tensor) -> Buffer:
        a = args[0]
        data = a.data.sum(axis=self.axis)
        return data

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        return (grad_output.reshape(self.ctx[0].shape),)