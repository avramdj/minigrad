from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np

from minigrad.device import DeviceManager


class Buffer(ABC):
    def __init__(self, data: Bufferable, device: str) -> None:
        super().__init__()
        assert device in DeviceManager.supported_devices
        self.device = device

    @abstractmethod
    def __getitem__(self, index: slice) -> Buffer:
        pass

    @abstractmethod
    def log(self) -> Buffer:
        pass

    @abstractmethod
    def exp(self) -> Buffer:
        pass

    @abstractmethod
    def transpose(self, axis=None) -> Buffer:
        pass

    @abstractmethod
    def sqrt(self) -> Buffer:
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def mean(self, axis=0) -> Buffer:
        pass

    @abstractmethod
    def size(
        self,
    ) -> int:
        pass


Bufferable = Union[int, float, list, np.ndarray, Buffer]
