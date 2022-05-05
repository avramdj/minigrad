import numpy as np
from .device import Device
from typing import Union, Iterable
from numbers import Number
from .device_function import *
from .utils import is_grad


class Tensor:
    """Tensor and its gradient"""

    def __init__(
        self, data: Union[list, np.ndarray, "Tensor"], device=None, requires_grad=True
    ):
        self.device = device if device else Device.CURRENT
        self.data = self._move(data, self.device)
        self.grad = None
        self.requires_grad = requires_grad
        self._function = None
        self._parents = []

    def backward(self):
        """
        Computes the gradients of `self` with respect to each node `x` and fills `x.grad`
        """
        visited = set()
        topo_nodes: list[Tensor] = []

        def topo_dag(node: Tensor):
            if node not in visited:
                visited.add(node)
                for p in node._parents:
                    topo_dag(p)
                topo_nodes.append(node)

        topo_dag(self)
        for v in reversed(topo_nodes):
            if not v.grad:
                v.grad = v.ones_like(v)
                v.grad.requires_grad = False
            if not v._function:
                continue
            nabla = v._function.backward(v.grad.data)
            for grad, parent in zip(nabla, v._parents):
                if parent.requires_grad:
                    parent.grad = Tensor(
                        grad + (parent.grad.data if parent.grad else 0)
                    )
                pass

    def _backward(self):
        """
        Recursive implementation of backward, test?
        """
        if not self.grad:
            self.grad = self.ones_like(self)
            self.grad.requires_grad = False
        if not self._function:
            return
        nabla = self._function.backward(self.grad.data)
        for grad, parent in zip(nabla, self._parents):
            parent.grad = Tensor(grad + (parent.grad.data if parent.grad else 0))
            parent.backward()

    @property
    def shape(self):
        return self.data.shape

    def detach(self):
        return self.__class__(self.data)

    def numpy(self):
        return self.detach().data

    def zero_grad(self):
        self.grad = None
        self._parents = []
        self._function = None

    def dot(self, other: "Tensor"):
        return self._binary_op(self, other, Dot)

    def log(self):
        return self._unary_op(self, Log)

    def exp(self):
        return self._unary_op(self, Exp)

    def tanh(self):
        return self._unary_op(self, Tanh)

    def sqrt(self):
        return self._unary_op(self, Sqrt)

    def relu(self):
        return self._unary_op(self, ReLU)

    def flatten(self, start_dim=0):
        return self._unary_param_op(self, start_dim, Flatten)

    def sum(self, axis=None):
        # assert axis is not None or len(self.shape) == 1, ("Must provide axis for element-wise "
        #                                                   "sum when dims > 1")
        return self._unary_param_op(self, axis, Sum)

    def mean(self, axis=None):
        # assert axis is not None or len(self.shape) == 1, ("Must provide axis for element-wise "
        #                                                   "sum when dims > 1")
        return self._unary_param_op(self, axis, Mean)

    def max(self, axis=None):
        return self._unary_param_op(self, axis, Max)

    def min(self, axis=None):
        return self._unary_param_op(self, axis, Min)

    def clip(self, a, b):
        ret = self.copy()
        ret.data.clip(a, b)
        return ret

    def copy(self):
        ret = self.__class__(self.data)
        ret.requires_grad = self.requires_grad
        ret._function = self._function
        ret._parents = self._parents
        if self.grad:
            ret.grad = self.grad.copy()
        return ret

    @property
    def T(self):
        return self._unary_op(self, Transpose)

    def __neg__(self):
        return self._unary_op(self, Neg)

    def __add__(self, other: "Tensor"):
        return self._binary_op(self, other, Add)

    def __radd__(self, other: "Tensor"):
        return self._binary_op(other, self, Add)

    def __iadd__(self, other):
        return self._binary_op(self, other, Add, inplace=True)

    def __sub__(self, other: "Tensor"):
        return self._binary_op(self, other, Sub)

    def __rsub__(self, other: "Tensor"):
        return self._binary_op(other, self, Sub)

    def __isub__(self, other: "Tensor"):
        return self._binary_op(self, other, Sub, inplace=True)

    def __mul__(self, other: "Tensor"):
        return self._binary_op(self, other, Mul)

    def __rmul__(self, other: "Tensor"):
        return self._binary_op(other, self, Mul)

    def __truediv__(self, other: "Tensor"):
        z = other**-1.0
        return self._binary_op(self, z, Mul)

    def __matmul__(self, other: "Tensor"):
        return self._binary_op(self, other, MatMul)

    def __pow__(self, alpha, modulo=None):
        return self._unary_param_op(self, alpha, Pow)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return f"<Tensor {self.data.shape}>"

    def __getitem__(self, item):
        return Tensor(self.data[item], requires_grad=self.requires_grad)

    def add_axis(self, axis):
        self.data = self.data.add_axis(axis)
        return self

    @staticmethod
    def _binary_op(a: "Tensor", b: "Tensor", func: Function.__class__, inplace=False):
        if isinstance(b, Number):
            b = Tensor(np.full(a.shape, b, dtype=np.float32), requires_grad=False)
        if isinstance(a, Number):
            a = Tensor(np.full(b.shape, a, dtype=np.float32), requires_grad=False)
        func = func(a.data, b.data)
        res = Tensor(func.forward(), requires_grad=(a.requires_grad or b.requires_grad))
        if not a.requires_grad and not b.requires_grad or is_grad():
            if inplace:
                a.data = res.data
                return a
            return res
        if inplace:
            raise Exception(
                "Attempted inplace operation on a leaf variable, set requires_grad=False or disable "
                "gradients globally"
            )
        res._function = func
        res._parents = [a, b]
        return res

    @staticmethod
    def _unary_op(a: "Tensor", func: Function.__class__):
        func = func(a.data)
        res = Tensor(func.forward(), requires_grad=a.requires_grad)
        if not a.requires_grad or is_grad():
            return res
        res._function = func
        res._parents = [a]
        return res

    @staticmethod
    def _unary_param_op(a: "Tensor", alpha, func: Function.__class__):
        func = func(a.data, alpha)
        res = Tensor(func.forward(), requires_grad=a.requires_grad)
        if not a.requires_grad or is_grad():
            return res
        res._function = func
        res._parents = [a]
        return res

    @classmethod
    def ones(cls, shape: Union[int, Iterable[int]], device=None, requires_grad=True):
        device = device if device else Device.CURRENT
        return cls(np.ones(shape), device=device, requires_grad=requires_grad)

    @classmethod
    def zeros(cls, shape: Union[int, Iterable[int]], device=None, requires_grad=True):
        device = device if device else Device.CURRENT
        return cls(np.zeros(shape), device=device, requires_grad=requires_grad)

    @classmethod
    def rand(cls, shape: Union[int, Iterable[int]], device=None, requires_grad=True):
        device = device if device else Device.CURRENT
        return cls(np.random.rand(*shape), device=device, requires_grad=requires_grad)

    @classmethod
    def rand_kaiming(
        cls, shape: Union[int, Iterable[int]], device=None, requires_grad=True
    ):
        device = device if device else Device.CURRENT
        # return cls(np.random.rand(*shape)*np.sqrt(2./np.prod(shape)), device=device, requires_grad=requires_grad)
        bound = np.sqrt(1.0 / shape[0])
        return cls(
            np.random.uniform(-bound, bound, shape),
            device=device,
            requires_grad=requires_grad,
        )

    @classmethod
    def full(
        cls,
        shape: Union[int, Iterable[int]],
        a: Union[int, float],
        device=None,
        requires_grad=True,
    ):
        device = device if device else Device.CURRENT
        return cls(np.full(shape, a), device=device, requires_grad=requires_grad)

    @classmethod
    def eye(cls, size: int, device=None, requires_grad=True):
        device = device if device else Device.CURRENT
        return cls(np.eye(size), device=device, requires_grad=requires_grad)

    @classmethod
    def ones_like(cls, a: Union[np.ndarray, "Tensor"], device=None, requires_grad=True):
        device = device if device else a.device
        if isinstance(a, Tensor):
            a = a.data
        return cls(
            np.ones_like(a, dtype=np.float32), device, requires_grad=requires_grad
        )

    @classmethod
    def rand_like(cls, a: Union[np.ndarray, "Tensor"], device=None, requires_grad=True):
        device = device if device else a.device
        if isinstance(a, Tensor):
            a = a.data
        return cls(
            np.random.rand(a, dtype=np.float32), device, requires_grad=requires_grad
        )

    @classmethod
    def zeros_like(
        cls, a: Union[np.ndarray, "Tensor"], device=None, requires_grad=True
    ):
        device = device if device else a.device
        if isinstance(a, Tensor):
            a = a.data
        return cls(
            np.zeros_like(a, dtype=np.float32), device, requires_grad=requires_grad
        )

    @staticmethod
    def _move(data, device) -> np.ndarray:
        if isinstance(data, Number):
            data = [data]
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            return Device[device].array(data)
        if isinstance(data, Tensor):
            return data.data
        if isinstance(data, Device[device]):
            return data
        raise RuntimeError(f"Unknown data type {type(data)}")


class Parameter(Tensor):
    def __init__(self, data: Union[list, np.ndarray, "Tensor"], **kwargs):
        super().__init__(data, **kwargs)


class Variable(Tensor):
    def __init__(self, data: Union[list, np.ndarray, "Tensor"], **kwargs):
        super().__init__(data, **kwargs)
        self.requires_grad = False
