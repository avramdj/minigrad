import numpy as np


class CPUDevice(np.ndarray):
    """
    The CPU device is a numpy interface
    """
    def log(self):
        return np.log(self)

    def pow(self, a):
        return np.power(self, a)

    def exp(self):
        return np.exp(self)

    def to_cpu(self):
        return self

    @staticmethod
    def array(data):
        return np.array(data).view(CPUDevice)
