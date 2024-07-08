import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

import minigrad
from minigrad import nn
from minigrad.data import DataLoader

n_classes = 10
digits = datasets.load_digits(n_class=n_classes)

X_train, X_test, y_train, y_test = train_test_split(digits.images, digits.target, train_size=0.8)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=0.8)
X_train /= 16.0
X_validation /= 16
X_test /= 16.0


class MnistClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, z_size=64):
        super().__init__()
        self.input_shape = input_shape
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
        with minigrad.no_grad():
            probs = self(x)
        return np.argmax(probs.data, axis=1)


shape = X_train[0].shape
batch_size = 64
epochs = 200

np.seterr(all="raise")

model = MnistClassifier(shape, n_classes)
optimizer = minigrad.optim.Adam(model.params(), learning_rate=1e-3)
criterion = nn.losses.MSE(n_classes)
train_loader = DataLoader(X_train, y_train, batch_size=batch_size, tensors=True)
losses = []

for i in range(epochs):
    loss = 0
    for x, gt in train_loader.get():
        outputs = model(x)
        cte = criterion(gt, outputs)
        cte.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss += cte.data.item()
    train_preds = model.predict(X_train)
    train_acc = metrics.accuracy_score(y_train, train_preds)
    validation_preds = model.predict(X_validation)
    validation_acc = metrics.accuracy_score(y_validation, validation_preds)
    print(
        f"Epoch {i+1:{len(str(epochs))}}/{epochs}, Loss: {loss:.3f}"
        f", Train accuracy: {train_acc*100:.1f}%"
        f", Validation accuracy: {validation_acc * 100:.1f}%"
    )
    losses.append(loss)

test_preds = model.predict(X_test)
test_acc = metrics.accuracy_score(y_test, test_preds)
print(f"Test accuracy: {test_acc*100:.1f}%")

plt.plot(losses)
plt.show()
