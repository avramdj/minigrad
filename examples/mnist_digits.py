import numpy as np
import minigrad
import minigrad as mg
from minigrad import nn
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm

digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.images, digits.target, train_size=0.8)
X_train /= 256.
X_test /= 256.


class MnistClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, z_size=128):
        super().__init__()
        self.input_shape = input_shape
        flatten_shape = input_shape[0] * input_shape[1]
        self.flatten = nn.layers.Flatten()
        self.linear1 = nn.layers.Linear(input_size=flatten_shape, output_size=z_size)
        self.activation = nn.layers.ReLU()
        self.linear2 = nn.layers.Linear(input_size=z_size, output_size=num_classes)
        self.softmax = nn.layers.Softmax()
        self._register_modules([self.linear1, self.linear2])

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

    def predict(self, x):
        with minigrad.no_grad():
            probs = self(x)
        return np.argmax(probs.data)[0]


ohe = OneHotEncoder()
y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()
shape = X_train[0].shape
n_classes = 10

model = MnistClassifier(shape, n_classes)
optimizer = minigrad.optim.SGD(model.params(), learning_rate=1e-3)
losses = []
epochs = 100
batch_size = 32

for i in range(epochs):
    loss = 0
    preds = []
    for idx, (sample, gt) in enumerate(zip(X_train, y_train_ohe)):
        x = minigrad.Tensor(sample, requires_grad=False)
        gtt = mg.Tensor(gt, requires_grad=False)
        output = model(x)
        preds.append(np.argmax(output.data))
        cte = nn.losses.cross_entropy(output, gtt)
        cte.backward()
        if (idx+1) % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
        loss = loss + cte.data
    acc = metrics.accuracy_score(y_train, preds)
    print(f"Epoch {i}/{epochs}, Loss: {loss:.3f}, Accuracy: {acc:.3f}")
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss)

plt.plot(losses)
plt.show()