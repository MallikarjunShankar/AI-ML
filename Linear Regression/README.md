# Week 1 — Linear Regression (NumPy Implementation)

## Overview

This project implements **Linear Regression from scratch using NumPy**.
The objective is to understand the **core mechanics of supervised learning**, including:

* model formulation
* loss functions
* gradient descent
* parameter optimization
* visualization of results

The implementation avoids machine learning libraries such as **scikit-learn** to focus on the mathematical and algorithmic foundations of the model.

The project generates a synthetic dataset, trains a regression model using gradient descent, and visualizes both the training data and the learned regression line.

---

# Problem Statement

The goal is to learn a function that approximates the relationship between input and output variables.

The true relationship used to generate the dataset is:

```
y = 4x + 3 + noise
```

Where:

* `x` = input feature
* `y` = output variable
* `noise` = random Gaussian noise simulating real-world data imperfections

The regression model attempts to learn:

```
y = wx + b
```

Where:

* `w` = slope (weight)
* `b` = intercept (bias)

---

# Dataset Generation

The dataset is generated using NumPy:

* 100 samples
* values randomly distributed
* Gaussian noise added to simulate real measurements

This allows the training algorithm to recover the underlying relationship.

Example generation logic:

```
X = np.random.rand(100,1) * 10
noise = np.random.randn(100,1)
y = 4 * X + 3 + noise
```

---

# Model Components

The implementation contains the following core components.

### 1. Prediction Function

Computes the predicted output.

```
y_pred = w * X + b
```

This function applies the regression equation to all data points.

---

### 2. Loss Function

The model uses **Mean Squared Error (MSE)**.

```
Loss = mean((y_pred - y_true)^2)
```

Purpose:

* measure prediction error
* provide a quantity that gradient descent minimizes

---

### 3. Gradient Computation

Gradients determine how parameters should change to reduce the loss.

```
dw = (2/n) * Σ x(y_pred - y)
db = (2/n) * Σ (y_pred - y)
```

These derivatives guide the optimization process.

---

### 4. Gradient Descent

Parameters are updated iteratively.

```
w = w - learning_rate * dw
b = b - learning_rate * db
```

Training proceeds for a fixed number of **epochs**.

---

# Training Process

The training loop performs the following steps repeatedly:

1. Predict outputs
2. Compute loss
3. Calculate gradients
4. Update parameters
5. Record loss history

Over time, the loss should decrease as the model converges.

---

# Visualization

Two visualizations are produced.

### Training Data

A scatter plot of the generated dataset.

```
plt.scatter(X, y)
```

---

### Regression Fit

A plot of the predicted regression line.

```
plt.plot(X, y_pred)
```

This shows how well the learned model approximates the data.

---

### Loss Curve

The loss values recorded during training are plotted to confirm that the model converges.

```
plt.plot(loss_history)
```

A decreasing curve indicates successful optimization.

---

# Expected Results

After training:

* the loss should decrease over time
* the regression line should closely match the dataset
* the learned parameters should approximate the true values

Example:

```
True model:
y = 4x + 3

Learned parameters:
w ≈ 4
b ≈ 3
```

