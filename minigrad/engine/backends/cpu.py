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
        return np.maximum(self, 0)

    def to_cpu(self):
        return self

    def add_axis(self, axis):
        return np.expand_dims(self, axis)

    def ones(self):
        return np.ones_like(self)

    def zeros(self):
        return np.zeros_like(self)

    def clip(self, a, b):
        return np.clip(self, a, b)

    def amax_(self, axis, keepdims=False):
        return np.amax(self, axis=axis, keepdims=keepdims)

    def swapaxes_(self, a, b):
        return np.swapaxes(self, a, b)

    def max_(self, axis, keepdims=False):
        return self.amax_(axis, keepdims=keepdims)

    def flatten_(self, start_dim=0):
        return self.reshape(-1, np.prod(self.shape[start_dim:]))

    def apply(self, func, axis):
        return np.apply_along_axis(func, axis, self)

    def expand_dims_(self, n):
        return np.expand_dims(self, n)

    def broadcast_to_(self, shape):
        return np.broadcast_to(self, shape)

    @staticmethod
    def array(data):
        return np.array(data).view(CPUDevice)
