import cupy as cu


class CudaDevice(cu.ndarray):
    """
    The CUDA device is a CuPy interface
    """
    def log(self):
        return cu.log(self)

    def pow(self, a):
        return cu.power(self, a)

    def exp(self):
        return cu.exp(self)

    def to_cpu(self):
        return self.get()

    @staticmethod
    def array(data):
        return cu.asarray(data).view(CudaDevice)