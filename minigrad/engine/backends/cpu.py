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

    def sqrt(self):
        return np.sqrt(self)

    def tanh(self):
        return np.tanh(self)

    def relu(self):
        return np.maximum(self, 0.)

    def to_cpu(self):
        return self

    def add_axis(self, axis):
        return np.expand_dims(self, axis)

    def ones(self):
        return np.ones_like(self)

    def zeros(self):
        return np.zeros_like(self)

    @staticmethod
    def array(data):
        return np.array(data).view(CPUDevice)
