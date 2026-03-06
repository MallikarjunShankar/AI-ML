import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.rand(100,1) * 10
noise = np.random.randn(100,1)

y = 4 * X + 3 + noise

w = np.random.randn()
b = np.random.randn()

def predict(X, w, b):
    return w * X + b

def compute_loss(y_true, y_pred):
    n = len(y_true)
    loss = np.sum((y_pred - y_true) ** 2) / n
    return loss

def compute_gradient(X, y, y_pred):
    n = len(y)
    dw = (2/n) * np.sum(X * (y_pred - y))
    db = (2/n) * np.sum(y_pred - y)
    return dw, db

learning_rate = 0.001
epochs = 1000

loss_history = []

for epoch in range(epochs):
    y_pred = predict(X, w, b)
    loss = compute_loss(y, y_pred)

    dw, db = compute_gradient(X, y, y_pred)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    loss_history.append(loss)

    if epoch % 100 == 0:
        print("Epoch:", epoch, " Loss:", loss)

plt.plot(loss_history)

plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()

y_final = predict(X, w, b)

plt.scatter(X, y)
plt.plot(X, y_final, color="red")
plt.title("Linear Regression Fit")

plt.show()