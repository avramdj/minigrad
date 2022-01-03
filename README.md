# minigrad
Minigrad is an automatic tensor differentiation engine with a deep learning library on top of it.

- PyTorch-like API
- GPU accelerated (soon)

### Dependencies
- `numpy`

### How it works

The gradient of a tensor function `z` with respect to `x` can be computed using the [chain rule](https://en.wikipedia.org/wiki/Chain_rule).

<img src="https://gcdn.pbrd.co/images/hKCJBEtyQ79h.png?o=1" width="20%"/>

This property of differentiation allows us to compute the gradient by dynamically building a
[directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) of the operations that produced `z` and visiting it in reverse topological order.

### Simple example

Minimizing the following real-valued function:

<!-- <img src="https://gcdn.pbrd.co/images/Bokf3btMl72H.png?o=1" width="45%"/> -->

<img src="https://gcdn.pbrd.co/images/HcMzdob6yUhg.png?o=1" width="35%"/>

which has a local minima at `x = -20` where `z(x) = 500`

```py
import minigrad
from minigrad import Tensor
import matplotlib.pyplot as plt

minigrad.set_device("cpu")

history = []
learning_rate = 1e-4
epochs = 300

x = Tensor([2]) # initialize 1-D tensor [2]

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

<img src="https://gcdn.pbrd.co/images/NR9J9XKnN6ER.png?o=1" width="45%"/>


### The Neural Network module
Solving [MNIST](https://en.wikipedia.org/wiki/MNIST_database) shouldn't be a problem once you have an autograd engine, but it's even easier with a neural network library

...hence `minigrad.nn`

```py
import minigrad
from minigrad import nn
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from minigrad.data import DataLoader
import numpy as np

n_classes = 10
digits = datasets.load_digits(n_class=n_classes)

X_train, X_test, y_train, y_test = train_test_split(digits.images, digits.target, train_size=0.8)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=0.8)
X_train /= 16.
X_validation /= 16
X_test /= 16.


class MnistClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, z_size=64):
        super().__init__()
        self.input_shape = input_shape
        # child modules like this are automatically registered in the parent module
        # if they have trainable params
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size=np.prod(input_shape), output_size=z_size)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(input_size=z_size, output_size=z_size)
        self.linear3 = nn.Linear(input_size=z_size, output_size=num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = minigrad.Tensor(x, requires_grad=False)
        with minigrad.no_grad(): # disables DAG construction for gradients
            probs = self(x)
        return np.argmax(probs.data, axis=1)


shape = X_train[0].shape
batch_size = 64
epochs = 50

model = MnistClassifier(shape, n_classes)

# instantiate our optimizer with model params
optimizer = minigrad.optim.Adam(model.params(), learning_rate=1e-3)

# since we're not passing the ground truth as a one hot array, we better specify n_classes
criterion = nn.losses.MSE(n_classes)

# wraps our data in batches nicely
train_loader = DataLoader(X_train, y_train, batch_size=batch_size, tensors=True)
losses = []

for i in range(epochs):
    total_loss = 0
    for x, gt in train_loader.get():
        outputs = model(x) # forward pass
        loss = criterion(gt, outputs) # compute loss
        loss.backward() # calculate the gradients
        optimizer.step() # update weights
        optimizer.zero_grad() # reset gradients
        total_loss += loss.data.item()
    train_preds = model.predict(X_train)
    train_acc = metrics.accuracy_score(y_train, train_preds)
    validation_preds = model.predict(X_validation)
    validation_acc = metrics.accuracy_score(y_validation, validation_preds)
    print(f"Epoch {i:{len(str(epochs))}}/{epochs}, Loss: {total_loss:.3f}"
          f", Train accuracy: {train_acc*100:.1f}%"
          f", Validation accuracy: {validation_acc * 100:.1f}%")
    losses.append(loss)

test_preds = model.predict(X_test)
test_acc = metrics.accuracy_score(y_test, test_preds)
print(f"Test accuracy: {test_acc*100:.1f}%")

plt.plot(losses)
plt.show()
```
#### Output and loss
```
Epoch 50/50, Loss: 0.023, Train accuracy: 99.7%, Validation accuracy: 95.5%
Test accuracy: 97.8%
```
<img src="https://gcdn.pbrd.co/images/KFzSOQvnIucV.png?o=1" width="45%"/>
