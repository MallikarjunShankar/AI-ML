import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification (
    n_samples = 1000,
    n_features = 2,
    n_classes = 2,
    n_redundant = 0,
    random_state = 42
)

y = y.reshape(-1, 1)

np.random.seed(42)

weights = np.random.randn(2,1)
bias = 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def BinaryCrossEntropyLoss(y, y_pred):
    m = len(y)
    loss = - (1/m) * np.sum( y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
    return loss

learning_rate = 0.01
epochs = 1000

losses = []

for epoch in range(epochs):
    # linear model
    z = np.dot(X, weights) + bias

    # prediction
    y_pred = sigmoid(z)

    # loss
    loss = BinaryCrossEntropyLoss(y, y_pred)
    losses.append(loss)

    # gradients
    m = len(y)
    dw = (1/m) * np.dot(X.T, y_pred - y)
    db = (1/m) * np.sum(y_pred - y)

    # update
    weights -= learning_rate * dw
    bias -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}   Loss: {loss}")

def predict(X):
    z = np.dot(X, weights) + bias
    probs = sigmoid(z)
    return (probs >= 0.5).astype(int)

y_pred = predict(X)

accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy}")

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()