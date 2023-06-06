from __future__ import annotations

import cupy as cp

from minigrad.core.buffer import Buffer, Bufferable


class CUDABuffer(Buffer):
    def __init__(self, data: Bufferable, from_ops=False, dtype=None) -> None:
        super().__init__(data, "cuda")
        self._data: cp.ndarray
        if from_ops and isinstance(data, cp.ndarray):
            self._data = data
        else:
            if isinstance(data, CUDABuffer):
                self._data = data._data.copy()
            else:
                # yes, it does copy the data
                self._data = cp.array(data, dtype=dtype)

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
        return CUDABuffer(cp.log(self._data))

    def exp(self) -> CUDABuffer:
        return CUDABuffer(cp.exp(self._data))

    def transpose(self, axes=None) -> CUDABuffer:
        data = self._data.copy()
        return CUDABuffer(data.transpose(axes=axes))

    def sqrt(self) -> CUDABuffer:
        return CUDABuffer(cp.sqrt(self._data))

    def mean(self, axis=0) -> CUDABuffer:
        return CUDABuffer(cp.mean(self._data, axis=axis))

    def numpy(self) -> cp.ndarray:
        return self._data

    def size(self) -> int:
        return self._data.size

    def flatten(self, start_dim=0) -> CUDABuffer:
        assert start_dim >= 0 and start_dim <= len(
            self._data.shape
        ), "start_dim must be in range [0, len(shape)]"
        return CUDABuffer(self._data.reshape(*self._data.shape[:start_dim], -1))

    def reshape(self, shape) -> CUDABuffer:
        return CUDABuffer(self._data.reshape(shape))
        
    def tanh(self) -> CUDABuffer:
        return CUDABuffer(cp.tanh(self._data))

    def sum(self, axis=None) -> CUDABuffer:
        return CUDABuffer(cp.sum(self._data, axis=axis))

    # TODO: sum, etc
