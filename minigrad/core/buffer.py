from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, Tuple, Union

import numpy as np

from minigrad.device import DeviceManager


class RawBufferInterface(Protocol):
    @property
    def dtype(self) -> np.dtype:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        ...


class Buffer(ABC):
    def __init__(self, data: Bufferable, device: str) -> None:
        super().__init__()
        assert device in DeviceManager.supported_devices
        self.device = device
        self._data: RawBufferInterface

    # fmt: off
    @abstractmethod
    def __getitem__(self, index: slice) -> Buffer: ...

    @abstractmethod
    def log(self) -> Buffer: ...

    @abstractmethod
    def exp(self) -> Buffer: ...

    @abstractmethod
    def transpose(self, axis=None) -> Buffer: ...

    @abstractmethod
    def sqrt(self) -> Buffer: ...

    @abstractmethod
    def numpy(self) -> np.ndarray: ...

    @abstractmethod
    def mean(self, axis=0) -> Buffer: ...

    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def flatten(self, start_dim=0) -> Buffer: ...

    @abstractmethod
    def reshape(self, shape) -> Buffer: ...

    @abstractmethod
    def tanh(self) -> Buffer: ...

    @abstractmethod
    def sum(self, axis=None) -> Buffer: ...
    # fmt: on


    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype


Bufferable = Union[int, float, list, np.ndarray, Buffer]
