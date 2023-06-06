import numpy as np

from minigrad.core.backends import CPUBuffer, CUDABuffer
from minigrad.core.buffer import Buffer, Bufferable
from minigrad.core.types import MinigradType
from minigrad.device import DeviceManager


def create_buffer(data: Bufferable, device: str, dtype: MinigradType) -> Buffer:
    if isinstance(data, Buffer):
        if data.device != device:
            data = data.numpy()
            # raise ValueError(f"device mismatch: Buffer device: `{data.device}` vs specified device: `{device}`")
    if device == "cpu":
        return CPUBuffer(data, dtype=dtype)
    elif device == "cuda":
        return CUDABuffer(data, dtype=dtype)
    raise ValueError(f"device '{device}' must be one of {DeviceManager.supported_devices}")
