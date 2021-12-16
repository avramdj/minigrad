class Function:
    """ Function base for Add, Sub, etc... """
    pass


class Add(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a + self.b

    def backward(self, grad):
        ones = self.a.ones() * grad
        return [ones, ones]


class Sub(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a - self.b

    def backward(self, grad):
        d = self.a.ones() * grad
        return [d, d]


class Mul(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a * self.b

    def backward(self, grad):
        da = self.b * grad
        db = self.a * grad
        return [da, db]


class Pow(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a.pow(self.b)

    def backward(self, grad):
        da = self.a.pow(self.b - 1.0) * self.b * grad
        return [da]


class Dot(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.is_vector = len(self.b.shape) == 1

    def forward(self):
        if self.is_vector:
            self.b.shape = (self.b.shape[0], 1)
            res = self.a.dot(self.b)
            assert len(res.shape) == 2
            res.shape = (res.shape[0],)
            self.b.shape = (self.b.shape[0],)
            return res
        return self.a.dot(self.b)

    def backward(self, grad):
        if self.is_vector:
            grad.shape = (grad.shape[0], 1)
            self.b.shape = (self.b.shape[0], 1)
            da = grad.dot(self.b.T)
            db = self.a.T.dot(grad)
            db.shape = (db.shape[0],)
            grad.shape = (grad.shape[0],)
            self.b.shape = (self.b.shape[0],)
        else:
            da = grad.dot(self.b.T)
            db = self.a.T.dot(grad)
        return [da, db]


class Log(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.log()

    def backward(self, grad):
        da = self.a.pow(-1.0) * grad
        return [da]


class Exp(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.exp()

    def backward(self, grad):
        da = self.forward() * grad
        return [da]


class Sqrt(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.sqrt()

    def backward(self, grad):
        da = 0.5 * self.forward().pow(-1) * grad
        return [da]


class ReLU(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.relu()

    def backward(self, grad):
        da = (self.a > 0) * grad
        return [da]


class Flatten(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.flatten()

    def backward(self, grad):
        da = self.a.zeros() * grad
        return [da]


class Sum(Function):
    def __init__(self, a, axis):
        self.a = a
        self.axis = axis

    def forward(self):
        return self.a.sum(axis=self.axis)

    def backward(self, grad):
        da = self.a.ones() * grad
        return [da]


class Neg(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return -self.a

    def backward(self, grad):
        da = -self.a.ones() * grad
        return [da]
