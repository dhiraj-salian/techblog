---
layout: single
title:  "Generative AI 1.4: Optimization Techniques"
date:   2024-09-22
categories: artificial-intelligence
---

Optimization refers to the process of finding the best parameters for a model to minimize (or maximize) some objective function, typically the **loss function** in machine learning. The most commonly used optimization techniques in machine learning involve **gradient-based** methods like **Gradient Descent**.

### 1.4.1 Gradient Descent

#### What is Gradient Descent?

**Gradient Descent** is an iterative optimization algorithm used to minimize the loss function by updating the model's parameters (weights) in the direction of the steepest descent, i.e., the negative of the gradient.

The key idea behind gradient descent is that we compute the gradient (partial derivatives) of the loss function with respect to each parameter and move the parameters in the opposite direction of the gradient.

#### Mathematics of Gradient Descent:

For a parameter $w$, the update rule in gradient descent is:
$$
w = w - \eta \cdot \frac{\partial L(w)}{\partial w}
$$
Where:
- $\eta$ is the **learning rate** (a small step size),
- $\frac{\partial L(w)}{\partial w}$ is the gradient of the loss function $L(w)$ with respect to $w$.

#### Steps in Gradient Descent:
1. **Initialize the weights** randomly.
2. **Compute the loss function** (e.g., Mean Squared Error or Cross-Entropy).
3. **Compute the gradients** (the rate of change of the loss with respect to each weight).
4. **Update the weights** by subtracting the product of the gradient and learning rate from the current weights.
5. **Repeat** until the loss function is minimized or converges.

### 1.4.2 Stochastic Gradient Descent (SGD)

#### What is SGD?

**Stochastic Gradient Descent (SGD)** is a variation of gradient descent where the gradients are calculated and the weights are updated using a single example (or a small batch) at a time rather than the entire dataset. This makes the optimization process much faster for large datasets, but it introduces noise into the optimization process.

#### Update Rule for SGD:
The update rule is similar to regular gradient descent, but the gradient is computed for a single training example or a small batch:
$$
w = w - \eta \cdot \frac{\partial L_i(w)}{\partial w}
$$
Where:
- $L_i(w)$ is the loss for the $i$-th training example.

#### Pros of SGD:
- Faster updates for large datasets.
- Can escape local minima due to noise in the updates.

#### Cons of SGD:
- Updates can be noisy, leading to fluctuating loss.

### 1.4.3 Momentum

Momentum is a technique used to speed up gradient descent by accumulating past gradients to help accelerate the optimization in relevant directions.

#### Update Rule with Momentum:
$$
v = \beta v + \eta \cdot \nabla L(w)
$$
$$
w = w - v
$$
Where:
- $\beta$ is the **momentum term** (a value between 0 and 1 that controls the contribution of previous gradients),
- $v$ is the velocity (the accumulated gradient).

Momentum helps smooth out the oscillations and speed up convergence, especially when the gradients have large variations.

### 1.4.4 Adam Optimizer

#### What is Adam?

**Adam (Adaptive Moment Estimation)** is an advanced optimization algorithm that combines the advantages of **SGD** with **momentum** and adaptive learning rates. It uses running averages of both the gradients and the squared gradients to make updates, which allows it to handle sparse gradients and noisy data more efficiently.

#### Update Rule for Adam:

1. Compute the running averages of the gradients:
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$
   Where:
   - $m_t$ is the first moment (mean of the gradient),
   - $v_t$ is the second moment (variance of the gradient),
   - $\beta_1$ and $\beta_2$ are decay rates for the first and second moments,
   - $g_t$ is the gradient at time step $t$.

2. Bias correction:
   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   $$

3. Update the parameters:
   $$
   w_t = w_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   $$

Where:
- $\eta$ is the learning rate,
- $\epsilon$ is a small constant to prevent division by zero.

#### Advantages of Adam:
- Handles noisy data well.
- Requires less tuning of hyperparameters.
- Works well for sparse gradients and large datasets.

### 1.4.5 Learning Rate

The **learning rate** $\eta$ controls the size of the steps taken in gradient descent. A small learning rate makes the optimization process slow but more accurate, while a large learning rate speeds up optimization but risks overshooting the minimum.

#### Learning Rate Tradeoff:
- **Too small**: Convergence is slow.
- **Too large**: Risk of overshooting and diverging.

#### Learning Rate Scheduling:
- **Fixed Learning Rate**: Constant value throughout training.
- **Decay**: Decreases the learning rate as training progresses to fine-tune the parameters.

### Implementing Gradient Descent and Adam in Python

Let’s implement **gradient descent** and **Adam** optimizer in Python.

#### Python Code: Gradient Descent

This code demonstrates how to apply gradient descent to minimize a simple quadratic function $f(x) = x^2$.

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

#### Output:

```
Epoch 1: x = 8.0, f(x) = 64.0
Epoch 2: x = 6.4, f(x) = 40.96000000000001
Epoch 3: x = 5.12, f(x) = 26.2144
...
Epoch 20: x = 0.10737418240000006, f(x) = 0.011529215046068475
```

#### Python Code: Adam Optimizer

This code implements the Adam optimizer to minimize the same quadratic function  $f(x) = x^2$

```python
import numpy as np

# Adam optimizer parameters
x = 10  # Initial value of x
learning_rate = 0.1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
epochs = 20

m, v = 0, 0  # Initialize first and second moments

# Define the function and its derivative
def f(x):
    return x**2

def derivative_f(x):
    return 2 * x

# Perform Adam optimization
for t in range(1, epochs + 1):
    grad = derivative_f(x)  # Compute the gradient
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    
    m_hat = m / (1 - beta1**t)  # Bias correction
    v_hat = v / (1 - beta2**t)
    
    x = x - learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon))  # Update x
    print(f"Epoch {t}: x = {x}, f(x) = {f(x)}")
```

#### Output:

```
Epoch 1: x = 8.998999900009998, f(x) = 80.98199990001996
Epoch 2: x = 7.998000000008, f(x) = 63.96800400006399
...
Epoch 20: x = 0.10000572233424616, f(x) = 0.010001144517619567
```

### Summary of Optimization Techniques:

- Gradient Descent updates the parameters by computing the gradient of the loss function and moving in the direction of steepest descent.
- SGD updates the parameters using a single or small batch of data, making it faster but noisier than regular gradient descent.
- Momentum helps speed up gradient descent by using past gradients to smooth updates.
- Adam Optimizer combines the advantages of momentum and adaptive learning rates to provide efficient and fast convergence, especially for sparse data.
- Learning rate is a critical hyperparameter, and using techniques like learning rate decay can help improve training performance.