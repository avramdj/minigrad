class Function:
    """
    Function base for Add, Sub, etc...
    IMPORTANT: These functions work on raw data buffers, not Tensors
    """
    pass


def unbroadcast(a, shape):
    axdiff = len(a.shape) - len(shape)
    if axdiff <= 0:
        return a
    return a.sum(axis=tuple(range(axdiff)))


class Add(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a + self.b

    def backward(self, grad):
        return [unbroadcast(grad, self.a.shape), unbroadcast(grad, self.b.shape)]


class Sub(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a - self.b

    def backward(self, grad):
        return [unbroadcast(grad, self.a.shape), unbroadcast(-grad, self.b.shape)]


class Mul(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a * self.b

    def backward(self, grad):
        da = self.b * grad
        db = self.a * grad
        return [unbroadcast(da, self.a.shape), unbroadcast(db, self.b.shape)]


class Pow(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a.pow(self.b)

    def backward(self, grad):
        da = self.a.pow(self.b - 1.0) * self.b * grad
        return [unbroadcast(da, self.a.shape)]


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
        return [unbroadcast(da, self.a.shape), unbroadcast(db, self.b.shape)]


class MatMul(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def forward(self):
        return self.a @ self.b

    def backward(self, grad):
        da = grad @ self.b.swapaxes_(-2, -1)
        db = self.a.swapaxes_(-2, -1) @ grad
        return [unbroadcast(da, self.a.shape), unbroadcast(db, self.b.shape)]


class Log(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.log()

    def backward(self, grad):
        da = self.a.pow(-1.0) * grad
        return [unbroadcast(da, self.a.shape)]


class Exp(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.exp()

    def backward(self, grad):
        da = self.a.exp() * grad
        return [unbroadcast(da, self.a.shape)]


class Sqrt(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.sqrt()

    def backward(self, grad):
        da = 0.5 * self.forward().pow(-1) * grad
        return [unbroadcast(da, self.a.shape)]


class Tanh(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.tanh()

    def backward(self, grad):
        da = (1 - self.a.tanh()**2) * grad
        return [unbroadcast(da, self.a.shape)]


class ReLU(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.relu()

    def backward(self, grad):
        da = (self.a > 0) * grad
        return [unbroadcast(da, self.a.shape)]


class Flatten(Function):
    def __init__(self, a, start_dim):
        self.a = a
        self.start_dim = start_dim

    def forward(self):
        return self.a.flatten_(start_dim=self.start_dim)

    def backward(self, grad):
        da = grad.reshape(self.a.shape)
        return [unbroadcast(da, self.a.shape)]


class Sum(Function):
    def __init__(self, a, axis):
        self.a = a
        self.axis = axis

    def forward(self):
        return self.a.sum(axis=self.axis, keepdims=True)

    # TODO: FIX
    def backward(self, grad):
        da = grad
        return [unbroadcast(da, self.a.shape)]


class Max(Function):
    def __init__(self, a, axis):
        self.a = a
        self.axis = axis

    def forward(self):
        return self.a.amax_(axis=self.axis, keepdims=True)

    def backward(self, grad):
        ones = self.a == grad
        div = ones.sum(axis=self.axis)
        da = ones * grad / div
        return [unbroadcast(da, self.a.shape)]


class Min(Function):
    """ bad """
    def __init__(self, a, axis):
        self.a = a
        self.axis = axis

    def forward(self):
        return self.a.amin(axis=self.axis, keepdim=True)

    def backward(self, grad):
        ones = self.a == grad
        div = ones.sum(axis=self.axis)
        da = ones * grad / div
        return [unbroadcast(da, self.a.shape)]


class Neg(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return -self.a

    def backward(self, grad):
        return [unbroadcast(-grad, self.a.shape)]


class Transpose(Function):
    def __init__(self, a):
        self.a = a

    def forward(self):
        return self.a.T

    # TODO: FINISH THIS
    def backward(self, grad):
        return [unbroadcast(grad.reshape(), self.a.shape)]
