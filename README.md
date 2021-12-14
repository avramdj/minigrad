# minigrad
Automatic tensor differentiation engine with a machine learning library on top of it.

- GPU accelerated
- PyTorch-like API

<!-- It's not quite [autograd](https://github.com/HIPS/autograd), definitely more than [micrograd](https://github.com/karpathy/micrograd) though, almost like [tinygrad](https://github.com/geohot/tinygrad)... -->

### How it works

The gradient of a tensor function `z` with respect to `x` can be computed using the [chain rule](https://en.wikipedia.org/wiki/Chain_rule).

<img src="https://gcdn.pbrd.co/images/hKCJBEtyQ79h.png?o=1" width="20%"/>

This property of differentiation allows us to compute the gradient by dynamically building a
[directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) of the operations and operands  that produced `z` and visiting it in reverse topological order.

### Simple example

Minimizing the following real-valued function:

<!-- <img src="https://gcdn.pbrd.co/images/Bokf3btMl72H.png?o=1" width="45%"/> -->

<img src="https://gcdn.pbrd.co/images/HcMzdob6yUhg.png?o=1" width="35%"/>

which has a local minima at `x = -20` where `z(x) = 500`

```py
import minigrad
from minigrad import Tensor
import matplotlib.pyplot as plt

minigrad.set_device("cuda")

history = []
learning_rate = 1e-4
epochs = 300

x = Tensor([2])

for i in range(epochs):
    z = (2*x + 50)**2 + x**2
    history.append(z.numpy()[0])
    z.backward() # compute gradient
    x -= x.grad * learning_rate # stochastic gradient descent, x.grad is dz/dx
    x.zero_grad() # reset gradient before next iteration
    z.zero_grad()

plt.figure()
plt.plot(history)
plt.show()
```

<img src="https://gcdn.pbrd.co/images/NR9J9XKnN6ER.png?o=1" width="35%"/>
