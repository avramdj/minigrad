import numpy as np
from device import Device
from typing import Union, Iterable
from numbers import Number
from function import Function, Add, Sub, Mul, Pow, Log, Exp, Neg


class Tensor:
    """ Tensor and its gradient """

    def __init__(self, data: Union[list, np.ndarray, "Tensor"], device=None, requires_grad=True):
        self.device = device if device else Device.CURRENT
        self._data = self._move(data, self.device)
        self._shape = self._data.shape
        self.grad = None
        self.requires_grad = requires_grad
        self._function = None
        self._parents = None

    def backward(self):
        if not self.grad:
            self.grad = self.ones_like(self)
            self.grad.requires_grad = False
        if not self._function:
            return
        nabla = self._function.backward()
        for derivative, parent in zip(nabla, self._parents):
            parent.grad = Tensor(self.grad._data * derivative)
            parent.backward()

    @property
    def shape(self):
        return self._shape

    def detach(self):
        return Tensor(self._data)

    def numpy(self):
        return self.detach()._data

    def zero_grad(self):
        self.grad = None
        self._parents = None
        self._function = None

    def __neg__(self):
        return self._unary_op(self, Neg)

    def __add__(self, other: "Tensor"):
        return self._binary_op(self, other, Add)

    def __sub__(self, other: "Tensor"):
        return self._binary_op(self, other, Sub)

    def __mul__(self, other: "Tensor"):
        return self._binary_op(self, other, Mul)

    def __truediv__(self, other: "Tensor"):
        z = other ** -1.0
        return self._binary_op(self, z, Mul)

    def __pow__(self, alpha, modulo=None):
        return self._unary_param_op(self, alpha, Pow)

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return f"<Tensor {self._data.shape}>"

    def __getitem__(self, item):
        return self._data[item]

    @staticmethod
    def _binary_op(a: "Tensor", b: "Tensor", func: Function.__class__):
        if isinstance(b, Number):
            b = Tensor(np.full(a.shape, b, dtype=np.float32))
        func = func(a._data, b._data)
        res = Tensor(func.forward())
        if not a.requires_grad:
            return res
        res._function = func
        res._parents = (a, b)
        return res

    @staticmethod
    def _unary_op(a: "Tensor", func: Function.__class__):
        func = func(a._data)
        res = Tensor(func.forward())
        if not a.requires_grad:
            return res
        res._function = func
        res._parents = (a,)
        return res

    @staticmethod
    def _unary_param_op(a: "Tensor", alpha, func: Function.__class__):
        func = func(a._data, alpha)
        res = Tensor(func.forward())
        if not a.requires_grad:
            return res
        res._function = func
        res._parents = (a,)
        return res

    @staticmethod
    def ones(shape: Union[int, Iterable[int]], device=Device.CURRENT):
        return Tensor(np.ones(shape), device=device)

    @staticmethod
    def full(shape: Union[int, Iterable[int]], a: Number, device=Device.CURRENT):
        return Tensor(np.full(shape, a), device=device)

    @staticmethod
    def ones_like(a: Union[np.ndarray, "Tensor"], device=None):
        device = device if device else a.device
        if isinstance(a, Tensor):
            a = a._data
        return Tensor(np.ones_like(a, dtype=np.float32), device)

    @staticmethod
    def _move(data, device) -> np.ndarray:
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            return data.astype(Device[device])
        if isinstance(data, Tensor):
            return data._data
        if isinstance(data, Device[device]):
            return data
        raise RuntimeError(f"Unknown data type {type(data)}")
