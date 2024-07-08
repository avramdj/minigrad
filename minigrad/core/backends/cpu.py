from __future__ import annotations

import numpy as np

from minigrad.core.buffer import Buffer, Bufferable


class CPUBuffer(Buffer):
    def __init__(self, data: Bufferable, from_ops=False, dtype=None) -> None:
        super().__init__(data, "cpu")
        self._data: np.ndarray
        if from_ops and isinstance(data, np.ndarray):
            self._data = data
        else:
            if isinstance(data, CPUBuffer):
                self._data = data._data.copy()
            else:
                # yes, it does copy the data
                self._data = np.array(data, dtype=dtype)

    def __getitem__(self, index: slice) -> CPUBuffer:
        return CPUBuffer(self._data[index])

    def __repr__(self) -> str:
        return f"CPUBuffer({self._data})"

    # fmt: off
    def __add__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data + other._data, from_ops=True)
    def __radd__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data + other._data, from_ops=True)
    def __sub__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data - other._data, from_ops=True)
    def __rsub__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data - other._data, from_ops=True)
    def __mul__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data * other._data, from_ops=True)
    def __rmul__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data * other._data, from_ops=True)
    def __truediv__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data / other._data, from_ops=True)
    def __rtruediv__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data / other._data, from_ops=True)
    def __pow__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data ** other._data, from_ops=True)
    def __rpow__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data ** other._data, from_ops=True)
    def __matmul__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data @ other._data, from_ops=True)
    def __rmatmul__(self, other: CPUBuffer) -> CPUBuffer: return CPUBuffer(self._data @ other._data, from_ops=True)
    def __neg__(self) -> CPUBuffer: return CPUBuffer(-self._data, from_ops=True)
    # fmt: on

    def log(self) -> CPUBuffer:
        return CPUBuffer(np.log(self._data))

    def exp(self) -> CPUBuffer:
        return CPUBuffer(np.exp(self._data))

    def transpose(self, axes=None) -> CPUBuffer:
        data = self._data.copy()
        return CPUBuffer(data.transpose(axes=axes))

    def sqrt(self) -> CPUBuffer:
        return CPUBuffer(np.sqrt(self._data))

    def mean(self, axis=0) -> CPUBuffer:
        return CPUBuffer(np.mean(self._data, axis=axis))

    def numpy(self) -> np.ndarray:
        return self._data

    def size(self) -> int:
        return self._data.size

    def flatten(self, start_dim=0) -> CPUBuffer:
        assert start_dim >= 0 and start_dim <= len(
            self._data.shape
        ), "start_dim must be in range [0, len(shape)]"
        return CPUBuffer(self._data.reshape(*self._data.shape[:start_dim], -1))

    def reshape(self, shape) -> CPUBuffer:
        return CPUBuffer(self._data.reshape(shape))
        
    def tanh(self) -> CPUBuffer:
        return CPUBuffer(np.tanh(self._data))

    def sum(self, axis=None) -> CPUBuffer:
        return CPUBuffer(np.sum(self._data, axis=axis))


    # TODO: sum, etc
