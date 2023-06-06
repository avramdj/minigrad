from __future__ import annotations

import subprocess
from typing import ClassVar, List, Optional


def cuda_is_available():
    try:
        subprocess.check_output(["nvcc", "--version"])
        return True
    except:
        return False


class DeviceManager:
    default: ClassVar[str] = "cpu"
    all_devices: ClassVar[List[str]] = ["cpu", "cuda"]
    current_device: ClassVar[str] = "cpu"
    supported_devices: ClassVar[List[str]] = None  # type: ignore

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("Device is a non-instantiable and non-subclassable class")

    @staticmethod
    def set_device(device: str):
        if device not in DeviceManager.all_devices:
            raise ValueError(f"{device} not found, must be one of {DeviceManager.all_devices}")
        if device not in DeviceManager.supported_devices:
            raise ValueError(f"{device} not supported, must be one of {DeviceManager.supported_devices}")
        DeviceManager.current_device = device

    @staticmethod
    def get_supported_devices() -> List[str]:
        if DeviceManager.supported_devices is not None:
            return DeviceManager.supported_devices
        supported_devices = ["cpu"]
        if cuda_is_available():
            supported_devices.append("cuda")
        return supported_devices

    @staticmethod
    def get_default_device():
        return DeviceManager.current_device

    @staticmethod
    def validate_device(device: Optional[str]) -> str:
        if device is None:
            return DeviceManager.current_device
        if device not in DeviceManager.supported_devices:
            raise ValueError(f"{device} not supported, must be one of {DeviceManager.supported_devices}")
        return device


DeviceManager.supported_devices = DeviceManager.get_supported_devices()
