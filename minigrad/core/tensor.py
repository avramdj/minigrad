from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

from minigrad.core import types
from minigrad.core.buffer import Buffer
from minigrad.core.buffer_manager import Bufferable, create_buffer
from minigrad.core.function import Add, Div, Exp, Function, Log, MatMul, Mean, Mul, Neg, Pow, Sqrt, Sub, Tanh, ReLU, Flatten, Sum
from minigrad.device import DeviceManager
from minigrad.core.context import is_grad

Tensorable = Union["Tensor", Bufferable, Buffer]
Shape = Tuple[int, ...]


class Tensor:
    def __init__(
        self,
        data: Tensorable,
        requires_grad: bool = False,
        device: Optional[str] = None,
        grad: Optional[Tensor] = None,
        grad_fn: Optional[Function] = None,
        dtype=None,
    ) -> None:
        self.device = DeviceManager.validate_device(device)
        self.dtype = dtype or types.float32
        self.data: Buffer = create_buffer(_get_raw_buffer(data), self.device, dtype=dtype)
        self.requires_grad: bool = requires_grad
        self.grad: Optional[Tensor] = grad
        self.grad_fn: Optional[Function] = grad_fn

    @staticmethod
    def binary_op(fn: Function, a: Tensor, b: Tensorable) -> Tensor:
        b = _as_tensor(b, device=a.device)
        requires_grad = a.requires_grad or b.requires_grad
        data = fn(a, b)
        grad_fn = fn if requires_grad else None
        return Tensor(data, requires_grad, a.device, grad_fn=grad_fn)

    @staticmethod
    def unary_op(fn: Function, a) -> Tensor:
        data = fn(a)
        grad_fn = fn if a.requires_grad else None
        return Tensor(data, a.requires_grad, a.device, grad_fn=grad_fn)

    # fmt: off
    def __repr__(self) -> str: return f"Tensor({self.data}, device={self.device}, requires_grad={self.requires_grad})"
    def __add__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(Add(), self, other)
    def __radd__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(Add(), self, other)
    def __sub__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(Sub(), self, other)
    def __rsub__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(Sub(), self, other)
    def __mul__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(Mul(), self, other)
    def __rmul__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(Mul(), self, other)
    def __truediv__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(Div(), self, other)
    def __rtruediv__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(Div(), self, other)
    def __pow__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(Pow(), self, other)
    def __rpow__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(Pow(), self, other)
    def __matmul__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(MatMul(), self, other)
    def __rmatmul__(self, other: Tensorable) -> Tensor: return Tensor.binary_op(MatMul(), self, other)
    def __neg__(self) -> Tensor: return Tensor.unary_op(Neg(), self)
    
    def log(self) -> Tensor: return Tensor.unary_op(Log(), self)
    def exp(self) -> Tensor: return Tensor.unary_op(Exp(), self)
    def sqrt(self) -> Tensor: return Tensor.unary_op(Sqrt(), self)
    def mean(self, axis=0) -> Tensor: return Tensor.unary_op(Mean(axis=axis), self)
    def tanh(self) -> Tensor: return Tensor.unary_op(Tanh(), self)
    def relu(self) -> Tensor: return Tensor.unary_op(ReLU(), self)
    def flatten(self, start_dim=0): return self.unary_op(Flatten(start_dim=start_dim), self)
    def sum(self, axis=None): return self.unary_op(Sum(axis=axis), self)

    #TODO: Check if this is should be TensorView and how to deal with slices
    def __getitem__(self, index: slice) -> Tensor: return TensorView(self, self.data[index], requires_grad=self.requires_grad, device=self.device)
    # fmt: on

    def T(self) -> Tensor:
        return self.transpose()

    def transpose(self) -> Tensor:
        return Tensor(self.data.transpose(), self.requires_grad, self.device)

    def detach(self) -> Tensor:
        return Tensor(self.data, False, self.device)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def numpy(self) -> np.ndarray:
        if self.grad_fn is not None:
            raise ValueError(
                "Tensor is part of a computation graph and cannot be converted to numpy. Detach first."
            )
        return self.data.numpy()

    def backward(self):
        if not self.requires_grad or not is_grad():
            return
        self.grad = Tensor(1.0, device=self.device)
        stack = [(self, self.grad)]
        while stack:
            tensor, grad = stack.pop()
            if tensor.grad_fn is not None:
                grads = tensor.grad_fn.backward(grad)
                for tensor, grad in zip(tensor.grad_fn.ctx, grads):
                    if grad is not None:
                        grad.requires_grad = False
                        stack.append((tensor, grad))
            else:
                # leaf node
                tensor.grad = grad

    def to(self, device: str) -> Tensor:
        if device == self.device:
            return self
        return Tensor(self.data, self.requires_grad, device)

    @classmethod
    def ones(cls, shape: Shape, device=None, requires_grad=True, dtype=None):
        return cls(np.ones(shape), device=device, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def zeros(cls, shape: Shape, device=None, requires_grad=True, dtype=None):
        return cls(np.zeros(shape), device=device, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def rand(cls, shape: Shape, device=None, requires_grad=True, dtype=None):
        return cls(np.random.rand(*shape), device=device, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def rand_kaiming(cls, shape: Shape, device=None, requires_grad=True, dtype=None):
        # return cls(np.random.rand(*shape)*np.sqrt(2./np.prod(shape)), device=device, requires_grad=requires_grad)
        bound = np.sqrt(1.0 / shape[0])
        return cls(
            np.random.uniform(-bound, bound, shape), device=device, requires_grad=requires_grad, dtype=dtype
        )

    @classmethod
    def full(cls, shape: Shape, a: Union[int, float], device=None, requires_grad=True, dtype=None):
        return cls(np.full(shape, a), device=device, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def eye(cls, size: int, device=None, requires_grad=True, dtype=None):
        return cls(np.eye(size), device=device, requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def ones_like(cls, a: Tensorable, device=None, requires_grad=True, dtype=None):
        if isinstance(a, Tensor):
            a = a.data
        return cls(np.ones_like(a), requires_grad=requires_grad, device=device, dtype=dtype)

    @classmethod
    def rand_like(cls, a: Tensorable, device=None, requires_grad=True, dtype=None):
        if isinstance(a, Tensor):
            a = a.data
        return cls(np.random.rand(a), requires_grad=requires_grad, device=device, dtype=dtype)

    @classmethod
    def zeros_like(cls, a: Tensorable, device=None, requires_grad=True, dtype=None):
        if isinstance(a, Tensor):
            a = a.data
        return cls(np.zeros_like(a), requires_grad=requires_grad, device=device, dtype=dtype)


class Parameter(Tensor):
    def __init__(self, data: Tensorable, **kwargs):
        super().__init__(data, **kwargs)


class Variable(Tensor):
    def __init__(self, data: Tensorable, **kwargs):
        super().__init__(data, **kwargs)
        self.requires_grad = False


class TensorView(Tensor):
    def __init__(self, original: Union[TensorView, Tensor], data: Buffer, requires_grad: bool, device: str):
        super().__init__(0, requires_grad=requires_grad, device=device)
        self.data = data
        self.original = original


def _as_tensor(data: Tensorable, requires_grad=False, device="cpu") -> Tensor:
    if isinstance(data, Tensor):
        if requires_grad:
            raise ValueError("cannot set requires_grad on existing tensor from `_as_tensor`")
        if device != data.device:
            raise ValueError("cannot change device on existing tensor from `_as_tensor`")
        return data
    return Tensor(data, requires_grad, device)


def _get_raw_buffer(data: Tensorable) -> Bufferable:
    if isinstance(data, Tensor):
        return data.data
    return data
