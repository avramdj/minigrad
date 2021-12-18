import numpy as np
import minigrad
import minigrad as mg
from minigrad import nn
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from minigrad.data import DataLoader
from tqdm import tqdm

digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.images, digits.target, train_size=0.8)
X_train /= 16.
X_test /= 16.


class MnistClassifier(nn.Module):
    def __init__(self, input_shape, num_classes, z_size=100):
        super().__init__()
        self.input_shape = input_shape
        self.flatten = nn.layers.Flatten()
        self.linear1 = nn.layers.Linear(input_size=np.prod(input_shape), output_size=z_size)
        self.activation = nn.layers.ReLU()
        self.linear2 = nn.layers.Linear(input_size=z_size, output_size=num_classes)
        # self.linear3 = nn.layers.Linear(input_size=z_size, output_size=num_classes)
        self.softmax = nn.layers.Softmax()

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
        return np.argmax(probs.data)


ohe = OneHotEncoder()
shape = X_train[0].shape
n_classes = 10
batch_size = 32
epochs = 1000

model = MnistClassifier(shape, n_classes)
optimizer = minigrad.optim.SGD(model.params(), learning_rate=1e-3)
criterion = nn.losses.CrossEntropy(n_classes)
train_loader = DataLoader(X_train, y_train, batch_size=batch_size, tensors=True)
losses = []
try:
    for i in range(epochs):
        loss = 0
        preds = []
        for x, gt in train_loader.get():
            outputs = model(x)
            cte = criterion(outputs, gt)
            cte.backward()
            loss += cte.data.item()
            optimizer.step()
            optimizer.zero_grad()
        with mg.no_grad():
            preds = model(mg.Tensor(X_train)).data
        preds = [np.argmax(x) for x in preds]
        acc = metrics.accuracy_score(y_train, preds)
        print(f"Epoch {i}/{epochs}, Loss: {loss:.3f}, Accuracy: {acc*100:.1f}%")
        losses.append(loss)
except KeyboardInterrupt:
    pass

plt.plot(losses)
plt.show()
