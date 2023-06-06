from __future__ import annotations

import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda import autoinit  # just importing this initializes pycuda
from pycuda import cumath

from minigrad.engine.buffer import Buffer, Bufferable


class CUDABuffer(Buffer):
    def __init__(self, data: Bufferable, from_ops=False) -> None:
        super().__init__(data, "cuda")
        self._data: gpuarray.GPUArray
        if from_ops:
            self._data = data
        else:
            if isinstance(data, CUDABuffer):
                self._data = data._data.copy()
            else:
                self._data = gpuarray.to_gpu(np.array(data))

    def __getitem__(self, index: slice) -> CUDABuffer:
        return CUDABuffer(self._data[index])

    def __repr__(self) -> str:
        return f"CUDABuffer({self._data})"

    # fmt: off
    def __add__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data + other._data, from_ops=True)
    def __radd__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data + other._data, from_ops=True)
    def __sub__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data - other._data, from_ops=True)
    def __rsub__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data - other._data, from_ops=True)
    def __mul__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data * other._data, from_ops=True)
    def __rmul__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data * other._data, from_ops=True)
    def __truediv__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data / other._data, from_ops=True)
    def __rtruediv__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data / other._data, from_ops=True)
    def __pow__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data ** other._data, from_ops=True)
    def __rpow__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data ** other._data, from_ops=True)
    def __matmul__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data @ other._data, from_ops=True)
    def __rmatmul__(self, other: CUDABuffer) -> CUDABuffer: return CUDABuffer(self._data @ other._data, from_ops=True)
    def __neg__(self) -> CUDABuffer: return CUDABuffer(-self._data, from_ops=True)
    # fmt: on

    def log(self) -> CUDABuffer:
        return CUDABuffer(self._data.log())

    def exp(self) -> Buffer:
        return CUDABuffer(self._data.log())

    def transpose(self, axes=None) -> Buffer:
        data = self._data.copy()
        return CUDABuffer(data.transpose(axes=axes))

    def sqrt(self) -> Buffer:
        return CUDABuffer(cumath.sqrt(self._data))

    def to_numpy(self) -> np.ndarray:
        return self._data.get()

    def mean(self, axis=0) -> Buffer:
        # TODO: ADD AXIS
        sum = gpuarray.sum(self._data)
        mean = sum / self._data.shape[axis]
        return CUDABuffer(mean)

    def size(self) -> int:
        return self._data.size
