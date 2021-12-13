import os
import backends


class Device:
    DEFAULT = os.environ.get("_MINIGRAD_DEVICE", "cpu")
    CURRENT = DEFAULT
    devices = {
        "cpu": backends.cpu.CPUDevice,
        "cuda": backends.cuda.CudaDevice
    }

    def __class_getitem__(cls, item: str):
        device = cls.devices.get(item.lower(), None)
        if not device:
            raise NotImplementedError(f"Device {item} not found")
        return device

    @staticmethod
    def set_device(device_name: str):
        Device.CURRENT = device_name

    @staticmethod
    def to(data, device):
        return Device.devices.get(device, Device.CURRENT).array(data)