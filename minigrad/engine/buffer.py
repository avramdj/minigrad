from abc import ABC

import minigrad

_DEVICES = ["cpu", "cuda"]


# create_buffer is a function that takes in data and device and returns a subclass of Buffer depending on the device
def create_buffer(data, device: str):
    if device == "cpu":
        return CPUBuffer(data)
    elif device == "cuda":
        return CUDABuffer(data)


class Buffer(ABC):
    def __init__(self, data, device: str) -> None:
        super().__init__()
        if device not in _DEVICES:
            raise ValueError(f"device must be one of {_DEVICES}")
        if device == "cuda" and not minigrad.utils.cuda.is_available():
            raise ValueError("cuda device is not available")


class CPUBuffer(Buffer):
    def __init__(self, data) -> None:
        super().__init__(data, "cpu")
        self.data = data

    def __repr__(self) -> str:
        return f"CPUBuffer({self.data})"


class CUDABuffer(Buffer):
    def __init__(self, data) -> None:
        super().__init__(data, "cuda")

    def __repr__(self) -> str:
        return f"CUDABuffer({self.data})"
