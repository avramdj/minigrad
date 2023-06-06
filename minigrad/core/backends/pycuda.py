from __future__ import annotations

from typing import Tuple

import numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from minigrad.core.buffer import Buffer, Bufferable
from minigrad.core.types import _ctype_map


class PyCudaBuffer(Buffer):
    def __init__(self, data: Bufferable, _no_copy=False) -> None:
        super().__init__(data, "cuda")
        self._data: gpuarray.GPUArray
        if _no_copy:
            self._data = data
        else:
            if isinstance(data, PyCudaBuffer):
                self._data = data._data.copy()
            else:
                self._data = gpuarray.to_gpu(np.array(data))

    def __getitem__(self, index: slice) -> PyCudaBuffer:
        return PyCudaBuffer(self._data[index])

    def __repr__(self) -> str:
        return f"PyCudaBuffer({self._data})"

    # fmt: off
    def __add__(self, other: PyCudaBuffer) -> PyCudaBuffer: return PyCudaBuffer(self._data + other._data, _no_copy=True)
    def __radd__(self, other: PyCudaBuffer) -> PyCudaBuffer: return PyCudaBuffer(self._data + other._data, _no_copy=True)
    def __sub__(self, other: PyCudaBuffer) -> PyCudaBuffer: return PyCudaBuffer(self._data - other._data, _no_copy=True)
    def __rsub__(self, other: PyCudaBuffer) -> PyCudaBuffer: return PyCudaBuffer(self._data - other._data, _no_copy=True)
    def __mul__(self, other: PyCudaBuffer) -> PyCudaBuffer: return PyCudaBuffer(self._data * other._data, _no_copy=True)
    def __rmul__(self, other: PyCudaBuffer) -> PyCudaBuffer: return PyCudaBuffer(self._data * other._data, _no_copy=True)
    def __truediv__(self, other: PyCudaBuffer) -> PyCudaBuffer: return PyCudaBuffer(self._data / other._data, _no_copy=True)
    def __rtruediv__(self, other: PyCudaBuffer) -> PyCudaBuffer: return PyCudaBuffer(self._data / other._data, _no_copy=True)
    def __pow__(self, other: PyCudaBuffer) -> PyCudaBuffer: return PyCudaBuffer(self._data ** other._data, _no_copy=True)
    def __rpow__(self, other: PyCudaBuffer) -> PyCudaBuffer: return PyCudaBuffer(self._data ** other._data, _no_copy=True)
    def __matmul__(self, other: PyCudaBuffer) -> PyCudaBuffer: return self.matmul(other)
    def __rmatmul__(self, other: PyCudaBuffer) -> PyCudaBuffer: return self.matmul(other)
    def __neg__(self) -> PyCudaBuffer: return PyCudaBuffer(-self._data, _no_copy=True)
    # fmt: on

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    def _generic_unary_math_kernel(
        self, cumath_function: str, block_dim: Tuple[int, ...] = (256, 1, 1)
    ) -> PyCudaBuffer:
        ctype = _ctype_map[self.dtype.type]
        cumath_function = cumath_function + "f" if ctype == "float" else cumath_function
        kernel_code = f"""
        #include <math.h>

        __global__ void generic_kernel({ctype}* input, {ctype}* output, int size)
        {{
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < size)
            {{
                output[tid] = {cumath_function}(input[tid]);
            }}
        }}
        """
        launch_kernel = SourceModule(kernel_code).get_function("generic_kernel")
        input_gpu = self._data
        output_gpu = cuda.mem_alloc_like(input_gpu)
        input_size = self._data.size
        grid_dim = ((input_size - 1) // block_dim[0] + 1, 1)
        launch_kernel(input_gpu, output_gpu, np.int32(input_size), block=block_dim, grid=grid_dim)

        return PyCudaBuffer(
            gpuarray.GPUArray(self._data.shape, self._data.dtype, gpudata=output_gpu), _no_copy=True
        )

    def matmul(self, other: PyCudaBuffer) -> PyCudaBuffer:
        return PyCudaBuffer(gpuarray.dot(self._data, other._data))

    def log(self) -> PyCudaBuffer:
        return self._generic_unary_math_kernel("log")

    def exp(self) -> PyCudaBuffer:
        return self._generic_unary_math_kernel("exp")

    def transpose(self, axes=None) -> PyCudaBuffer:
        data = self._data.copy()
        return PyCudaBuffer(data.transpose(axes=axes))

    def sqrt(self) -> PyCudaBuffer:
        return self._generic_unary_math_kernel("sqrt")

    def numpy(self) -> np.ndarray:
        return self._data.get()

    def mean(self, axis=0) -> PyCudaBuffer:
        # TODO: ADD AXIS
        sum = gpuarray.sum(self._data)
        mean = sum / self._data.shape[axis]
        return PyCudaBuffer(mean)

    def size(self) -> int:
        return self._data.size
