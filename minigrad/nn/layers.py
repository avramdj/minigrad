from minigrad.nn import Module
from minigrad import Tensor, Parameter


class Linear(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        if isinstance(input_size, tuple):
            input_size = input_size[0]
        if isinstance(output_size, tuple):
            output_size = output_size[0]
        self.input_size = input_size
        self.output_size = output_size
        self.matrix = Parameter.rand_kaiming((self.output_size, self.input_size), requires_grad=True)
        self.bias = Parameter.ones((self.output_size,), requires_grad=True)

    def forward(self, x):
        res = self.matrix.dot(x) + self.bias
        return res


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (1+(-x).exp())**(-1)


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.tanh()


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        e_x = x.exp()
        return e_x / e_x.sum()


class LogSoftmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - x.max()


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.flatten()
