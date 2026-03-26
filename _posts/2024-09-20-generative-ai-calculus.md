---
layout: single
title:  "Generative AI 1.2: Calculus"
date:   2024-09-20
categories: [deep-learning]
tags: [calculus, mathematics, foundations, generative-ai]
description: "Understand how derivatives, gradients, and backpropagation enable neural networks to learn from data"
author: Dhiraj Salian
---

# Calculus for Generative AI

> Every time a neural network learns, it's calculus in action. Derivatives tell the model which direction to adjust its errors—and that's how intelligence emerges from math.

Calculus is the engine that drives machine learning. Without it, models couldn't learn from mistakes. Here's how it works.

## Derivatives: Measuring Change

A **derivative** measures how a function changes as its input changes. In ML, this tells us how the loss changes as we adjust weights.

For $f(x) = x^2$:
$$f'(x) = \frac{d}{dx}(x^2) = 2x$$

This means: at any point $x$, the rate of change is $2x$.

### Partial Derivatives

Neural networks have thousands (or millions) of parameters. We need **partial derivatives**—how the function changes with respect to one variable at a time.

For $f(x, y) = 3x^2 + 2y$:
$$\frac{\partial f}{\partial x} = 6x, \quad \frac{\partial f}{\partial y} = 2$$

## Gradients: The Learning Direction

A **gradient** is a vector of all partial derivatives—it points in the direction of steepest increase:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \end{bmatrix}$$

For $f(x, y) = 3x^2 + 2y$:
$$\nabla f = \begin{bmatrix} 6x \\ 2 \end{bmatrix}$$

In neural networks, the gradient of the loss function tells us how to adjust every weight to reduce error.

## Chain Rule: Layer by Layer

The **chain rule** computes derivatives of composite functions—exactly what happens in deep networks where layers are stacked:

$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

**Example**: For $f(x) = (2x + 3)^2$:
- Let $g(x) = 2x + 3$
- $f'(g(x)) = 2g(x)$, $g'(x) = 2$
- $\frac{d}{dx}(2x + 3)^2 = 2(2x + 3) \cdot 2 = 4(2x + 3)$

This is exactly how **backpropagation** works—propagating errors backward through layers.

## Gradient Descent: The Optimization

**Gradient descent** minimizes loss by moving against the gradient:

$$w = w - \eta \cdot \nabla f(w)$$

Where $\eta$ is the learning rate—a small step size.

**Example**: If gradient is $0.5$ and learning rate is $0.1$:
$$w_{new} = w_{old} - 0.1 \cdot 0.5 = w_{old} - 0.05$$

## Backpropagation: Learning Algorithm

**Backpropagation** is the algorithm that makes neural networks learn:

1. **Forward pass**: Compute predictions
2. **Compute loss**: Compare predictions to targets
3. **Backward pass**: Propagate error using chain rule
4. **Update weights**: Use gradient descent

```python
import numpy as np

def f(x):
    return x**2

def derivative_f(x):
    return 2 * x

# Gradient descent on f(x) = x²
x = 10  # Start somewhere
learning_rate = 0.1

for i in range(20):
    grad = derivative_f(x)
    x = x - learning_rate * grad
    print(f"Epoch {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}")

# Output converges to x = 0, f(x) = 0
```

## The Bottom Line

- **Derivatives** measure change—core to understanding how models learn
- **Gradients** point the way—tell us which direction to adjust
- **Chain rule** enables deep learning—computes gradients through many layers
- **Gradient descent** optimizes—iteratively minimizes loss
- **Backpropagation** implements it—practical algorithm for neural networks

This is how neural networks learn from data. Every training iteration is calculus in action.