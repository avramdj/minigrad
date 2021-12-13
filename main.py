import minigrad
from minigrad import Tensor
import matplotlib.pyplot as plt
from tqdm import tqdm

shape = (512, 512)

minigrad.set_device("cuda")

x = Tensor.full(shape, 10)
y = Tensor.full(shape, 4)

lr = 1e-3

losses = []

for i in tqdm(range(500)):
    z = (x + 1) ** 2 + y ** 2
    losses.append(z.numpy()[0])
    z.backward()
    x -= x.grad * lr
    y -= y.grad * lr
    x.zero_grad()
    y.zero_grad()
    z.zero_grad()

plt.figure()
plt.plot(losses)
plt.show()
