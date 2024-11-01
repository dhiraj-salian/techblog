---
layout: single
title:  "Generative AI 1.2: Calculus"
date:   2024-09-20
categories: artificial-intelligence
---

Calculus plays a fundamental role in machine learning, especially in training neural networks. During training, the network adjusts its parameters (weights and biases) to minimize a loss function. This process is made possible by computing **derivatives** (rates of change), which tell us how small changes in inputs affect the output.

## 1.2.1 Derivatives

A **derivative** measures how a function changes as its input changes. In machine learning, we use derivatives to compute how much the loss function changes as we adjust the weights. This is the key to training neural networks.

### Derivative of a Simple Function:

Given a function \\(f(x)\\), the derivative of \\(f\\) with respect to \\(x\\), denoted as \\(f'(x)\\) or \\(\\frac{df}{dx}\\), tells us the rate of change of \\(f(x)\\) at any point \\(x\\).

### Example:
For the function \\(f(x) = x^2\\), the derivative is:
\\[
f'(x) = \\frac{d}{dx}(x^2) = 2x
\\]
This means that for any value of \\(x\\), the rate of change of \\(f(x)\\) is \\(2x\\).

### Partial Derivatives:

When working with functions of multiple variables (e.g., neural networks with many weights), we compute **partial derivatives**. A partial derivative shows how a function changes with respect to one variable, keeping the others constant.

For example, for the function \\(f(x, y) = 3x^2 + 2y\\), the partial derivatives are:
\\[
\\frac{\\partial f}{\\partial x} = 6x, \\quad \\frac{\\partial f}{\\partial y} = 2
\\]

## 1.2.2 Gradients

A **gradient** is a vector that contains all the partial derivatives of a function with respect to its inputs. The gradient points in the direction of the steepest increase in the function, and it’s crucial for optimization in neural networks.

### Gradient Formula:

If \\(f\\) is a function of multiple variables \\(x_1, x_2, \\dots, x_n\\), the gradient is the vector of partial derivatives:
\\[
\\nabla f(x_1, x_2, \\dots, x_n) = \\begin{bmatrix} \\frac{\\partial f}{\\partial x_1} \\\\ \\frac{\\partial f}{\\partial x_2} \\\\ \\vdots \\\\ \\frac{\\partial f}{\\partial x_n} \\end{bmatrix}
\\]

### Example:

For the function \\(f(x, y) = 3x^2 + 2y\\), the gradient is:
\\[
\\nabla f(x, y) = \\begin{bmatrix} \\frac{\\partial f}{\\partial x} \\\\ \\frac{\\partial f}{\\partial y} \\end{bmatrix} = \\begin{bmatrix} 6x \\\\ 2 \\end{bmatrix}
\\]

In neural networks, the gradient of the **loss function** with respect to the weights is computed during **backpropagation** to update the weights in the direction that minimizes the loss.

## 1.2.3 Chain Rule

The **chain rule** is used to compute the derivative of a composite function, i.e., a function made up of other functions. The chain rule allows us to compute how changes in one variable affect another through a series of intermediate steps.

### Chain Rule Formula:

If \\(f(g(x))\\) is a composite function, the chain rule states:
\\[
\\frac{d}{dx} f(g(x)) = f'(g(x)) \\cdot g'(x)
\\]

### Example:

Let’s say \\(f(x) = (2x + 3)^2\\). To compute the derivative using the chain rule:
- Let \\(g(x) = 2x + 3\\), and \\(f(g(x)) = g(x)^2\\).
- First, compute \\(f'(g(x)) = 2g(x)\\) and \\(g'(x) = 2\\).
- Applying the chain rule:
\\[
\\frac{d}{dx}(2x + 3)^2 = 2(2x + 3) \\cdot 2 = 4(2x + 3)
\\]

In neural networks, the chain rule is used to compute the gradients during **backpropagation**, where the gradient of the loss with respect to earlier layers depends on the gradients of later layers.

## 1.2.4 Gradient Descent

**Gradient Descent** is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent (the negative of the gradient). In the context of neural networks, gradient descent is used to update the model's weights and biases to minimize the loss function.

### Gradient Descent Update Rule:

For a function \\(f\\) with respect to a parameter \\(w\\), the weight update rule is:
\\[
w = w - \\eta \\cdot \\nabla f(w)
\\]
Where:
- \\(w\\) is the weight,
- \\(\\eta\\) is the **learning rate** (a small step size),
- \\(\\nabla f(w)\\) is the gradient of the loss function with respect to \\(w\\).

### Example:
If the gradient of the loss function is \\(\\nabla f(w) = 0.5\\) and the learning rate is \\(\\eta = 0.1\\), the weight update would be:
\\[
w_{\\text{new}} = w_{\\text{old}} - 0.1 \\cdot 0.5 = w_{\\text{old}} - 0.05
\\]

## 1.2.5 Backpropagation

**Backpropagation** is the process of computing the gradients of the loss function with respect to each weight in the network. It uses the chain rule to propagate the error from the output layer back through the hidden layers, allowing us to update the weights using gradient descent.

### Steps in Backpropagation:
1. **Forward Pass**: Compute the predicted output.
2. **Compute the Loss**: Calculate the difference between the predicted output and the actual target.
3. **Backward Pass**: Propagate the error backwards, computing gradients using the chain rule.
4. **Update Weights**: Use gradient descent to update the weights.

## Derivatives and Gradient Descent in Python

Here’s a simple implementation to compute derivatives and perform gradient descent on a quadratic function \\(f(x) = x^2\\).

### Python Code:
```python
import numpy as np

# Define the function and its derivative
def f(x):
    return x**2

def derivative_f(x):
    return 2 * x

# Gradient descent parameters
x = 10  # Initial value of x
learning_rate = 0.1
epochs = 20

# Perform gradient descent
for i in range(epochs):
    grad = derivative_f(x)  # Compute the gradient
    x = x - learning_rate * grad  # Update x
    print(f"Epoch {i+1}: x = {x}, f(x) = {f(x)}")
```

### Output:
```
Epoch 1: x = 8.0, f(x) = 64.0
Epoch 2: x = 6.4, f(x) = 40.96000000000001
Epoch 3: x = 5.12, f(x) = 26.2144
Epoch 4: x = 4.096, f(x) = 16.776704
Epoch 5: x = 3.2768, f(x) = 10.737090560000001
...
Epoch 20: x = 0.10737418240000006, f(x) = 0.011529215046068475

```
{: .no-copy}

## Summary:
- Derivatives measure how a function changes as its input changes. They are essential for computing gradients in neural networks.
- Gradients are vectors that point in the direction of the steepest increase of a function. In machine learning, they are used to update the model parameters.
- The chain rule is crucial for computing gradients in multi-layered neural networks.
- Gradient Descent is an optimization technique used to minimize the loss function by adjusting the weights in the direction of the negative gradient.
- Backpropagation computes the gradients of the loss with respect to each weight in the network, allowing us to update the weights and minimize the error.
