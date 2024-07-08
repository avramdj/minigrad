from minigrad import Parameter, Tensor
from minigrad.nn.module import Module


class Linear(Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.matrix = Parameter.rand_kaiming((self.input_size, self.output_size), requires_grad=True)
        self.bias = Parameter.ones((self.output_size,), requires_grad=True)

    def forward(self, x):
        res = x @ self.matrix + self.bias
        return res


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.relu()


class LeakyReLU(Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("No leaky relu yet...")

    def forward(self, x):
        pass


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (1 + (-x).exp()) ** (-1)


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
        exs = e_x.sum(axis=1)
        return e_x / exs


class LogSoftmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - x.max()


class Flatten(Module):
    def __init__(self, batched=True):
        super().__init__()
        self.batched = batched

    def forward(self, x):
        return x.flatten(start_dim=1 if self.batched else 0)
