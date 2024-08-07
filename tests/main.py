import matplotlib.pyplot as plt

import minigrad as mg
from minigrad import nn


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mg.Parameter([1])
        self.layer_1 = lambda x: (x + self.w) ** 2

    def forward(self, x):
        x = self.layer_1(x)
        return x


mg.set_device("cpu")

learning_rate = 1e-3
epochs = 5000

model = DummyNet()
optimizer = mg.optim.SGD(model.params(), learning_rate=learning_rate)
losses = []
X = mg.Tensor([5], requires_grad=False)

for i in range(epochs):
    out = model(X)
    out.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(out.data.item())

print(X)
print(model.w)
plt.figure()
plt.plot(losses)
plt.show()
