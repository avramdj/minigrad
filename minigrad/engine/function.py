import numpy as np

class Function:
    """ Function base for Add, Sub, etc... """
    pass


class Add(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a + self.b

    def backward(self):
        ones = np.ones_like(self.a)
        return np.array([ones, ones])


class Sub(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a - self.b

    def backward(self):
        d = np.ones_like(self.a)
        return np.array([d, d])


class Mul(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a * self.b

    def backward(self):
        da = self.b
        db = self.a
        return np.array([da, db])


class Pow(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a.pow(self.b)

    def backward(self):
        da = self.a.pow(self.b - 1.0) * self.b
        return np.array([da])


class Log(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.log()

    def backward(self):
        da = self.a.pow(-1.0)
        return np.array([da])


class Exp(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.exp()

    def backward(self):
        da = self.forward()
        return np.array([da])


class Neg(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return -self.a

    def backward(self):
        da = -np.ones_like(self.a)
        return np.array([da])
