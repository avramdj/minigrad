import numpy as np

from minigrad.device import DeviceManager
from minigrad.engine.backends.cpu import CPUBuffer
from minigrad.engine.backends.cuda import CUDABuffer
from minigrad.engine.buffer import Buffer, Bufferable


def create_buffer(data: Bufferable, device: str) -> Buffer:
    if isinstance(data, Buffer):
        if data.device != device:
            data = data.to_numpy()
            # raise ValueError(f"device mismatch: Buffer device: `{data.device}` vs specified device: `{device}`")
    if device == "cpu":
        return CPUBuffer(data)
    elif device == "cuda":
        return CUDABuffer(data)
    raise ValueError(f"device '{device}' must be one of {DeviceManager.supported_devices}")
