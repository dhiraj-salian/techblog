---
layout: single
title: "Simple Neural Network in Python with Math behind it"
date: 2024-09-23
categories: artificial-intelligence
---

Neural networks are one of the most powerful tools in machine learning, capable of recognizing patterns, classifying images, and even generating text. But what exactly is happening inside them? In this blog, we'll walk through a simple neural network example in Python and explore the detailed math behind each step in an approachable way.

## What Is a Neural Network?

At its core, a neural network is a series of layers of interconnected neurons. Each layer performs a linear transformation (like matrix multiplication), and then applies an activation function to introduce non-linearity, which helps the network learn complex patterns. Finally, the network uses a loss function to measure how far off its predictions are, and backpropagation is used to adjust the weights to minimize this loss.

We'll use a simple neural network example in Python with one hidden layer and a small dataset, and we'll walk through both the forward and backward passes.

## Python Code Example

Here’s a small neural network implemented in PyTorch:

```python
import torch
import torch.nn as nn

# Define a simple neural network with one hidden layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Input layer: 2 inputs, 2 hidden neurons
        self.fc2 = nn.Linear(2, 1)  # Output layer: 1 neuron (binary classification)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for hidden layer
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for output layer
        return x

# Create the model and define a sample input
model = SimpleNN()
input_data = torch.tensor([[1.0, 2.0]])  # Single input sample
output = model(input_data)
print("Output:", output)
```

This neural network consists of:
- **Input**: A vector with two values (e.g., features of a sample).
- **Hidden layer**: Two neurons with ReLU activation.
- **Output layer**: One neuron with a sigmoid activation function, for binary classification.

Now let’s explore the **math** behind each step in detail.

## Step-by-Step Math Behind the Neural Network

### 1. Layer 1: Linear Transformation

The first layer takes two inputs and transforms them using a weight matrix $$ W_1 $$ and a bias vector $$ b_1 $$. This is called a **linear transformation**:

$$
z_1 = W_1 \cdot x + b_1
$$

In our example, the input $$ x = \begin{bmatrix} 1 \\ 2 \end{bmatrix} $$ and suppose the initial weights and biases are:

$$
W_1 = \begin{bmatrix} 0.2 & 0.4 \\ 0.1 & 0.6 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0.3 \\ 0.5 \end{bmatrix}
$$

The output from the first layer is:

$$
z_1 = \begin{bmatrix} 0.2 & 0.4 \\ 0.1 & 0.6 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0.3 \\ 0.5 \end{bmatrix} = \begin{bmatrix} 1.3 \\ 1.8 \end{bmatrix}
$$

### 2. Layer 1: ReLU Activation

The ReLU activation function is applied element-wise to the output of the first layer:

$$
\text{ReLU}(z_1) = \max(0, z_1)
$$

Since both values are positive, the output remains the same:

$$
a_1 = \text{ReLU}(z_1) = \begin{bmatrix} 1.3 \\ 1.8 \end{bmatrix}
$$

### 3. Layer 2: Linear Transformation

Now, the result from the hidden layer is passed to the second (output) layer. We perform another linear transformation using the weights $$ W_2 $$ and bias $$ b_2 $$:

$$
z_2 = W_2 \cdot a_1 + b_2
$$

Suppose the weights and bias for the second layer are:

$$
W_2 = \begin{bmatrix} 0.7 & 0.9 \end{bmatrix}, \quad b_2 = 0.2
$$

Then the output is:

$$
z_2 = \begin{bmatrix} 0.7 & 0.9 \end{bmatrix} \cdot \begin{bmatrix} 1.3 \\ 1.8 \end{bmatrix} + 0.2 = 3.18
$$

### 4. Layer 2: Sigmoid Activation

The sigmoid activation function is applied to transform $$ z_2 $$ into a probability:

$$
\hat{y} = \frac{1}{1 + e^{-z_2}} = \frac{1}{1 + e^{-2.73}} = 0.938
$$

Thus, the predicted output of the network is 0.96, which is close to 1, meaning the network is confident the input belongs to the positive class.

## Backpropagation: Adjusting the Weights

After the forward pass, we calculate the loss (how wrong the network’s prediction is), and we adjust the weights using backpropagation.

### 1. Loss Calculation

We use binary cross-entropy loss for classification:

$$
L = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

If the true label $$ y = 1 $$, and $$ \hat{y} = 0.938 $$, the loss is:

$$
L = -[\log(0.938)] = 0.064
$$

### 2. Backpropagation: Computing Gradients

To adjust the weights, we compute the gradients of the loss with respect to each weight using the **chain rule** from calculus. Let’s go step by step for each layer.

#### Gradient of the Loss with Respect to $$ \hat{y} $$

First, we calculate how much the loss changes with respect to the predicted output $$ \hat{y} $$:

$$
\frac{\partial L}{\partial \hat{y}} = -\frac{1}{\hat{y}} + \frac{0}{1 - \hat{y}} = -\frac{1}{0.938} = -1.066
$$

#### Gradient of $$ \hat{y} $$ with Respect to $$ z_2 $$ (Output of the Final Layer)

Since $$ \hat{y} $$ is passed through a sigmoid function, the derivative of the sigmoid function is:

$$
\frac{\partial \hat{y}}{\partial z_2} = \hat{y}(1 - \hat{y}) = 0.938 \times (1 - 0.938) = 0.0582
$$

#### Gradient of the Loss with Respect to $$ z_2 $$

Now, using the chain rule, we multiply the above two values to get:

$$
\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial z_2} = -1.066 \times 0.0582 = -0.062
$$

#### Gradient of $$ z_2 $$ with Respect to $$ W_2 $$ (Second Layer Weights)

The value of $$ z_2 $$ is a linear function of $$ W_2 $$, and so the gradient of $$ z_2 $$ with respect to $$ W_2 $$ is the hidden layer output $$ a_1 $$:

$$
\frac{\partial z_2}{\partial W_2} = a_1 = \begin{bmatrix} 1.3 \\ 1.8 \end{bmatrix}
$$

#### Gradient of the Loss with Respect to $$ W_2 $$

Now, using the chain rule, we multiply:

$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \times \frac{\partial z_2}{\partial W_2} = -0.062 \times \begin{bmatrix} 1.3 \\ 1.8 \end{bmatrix} = \begin{bmatrix} -0.0806 \\ -0.1116 \end{bmatrix}
$$

This tells us how to adjust the weights of $$ W_2 $$ to reduce the loss.

### 3. Backpropagation to the Hidden Layer

#### Gradient of $$ z_2 $$ with Respect to $$ a_1 $$ (Output of the First Layer)

Next, we calculate how much the loss affects the hidden layer output $$ a_1 $$:

$$
\frac{\partial z_2}{\partial a_1} = W_2 = \begin{bmatrix} 0.7 \\ 0.9 \end{bmatrix}
$$

#### Gradient of the Loss with Respect to $$ a_1 $$

Using the chain rule:

$$
\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial z_2} \times \frac{\partial z_2}{\partial a_1} = -0.039936 \times \begin{bmatrix} 0.7 \\ 0.9 \end{bmatrix} = \begin{bmatrix} -0.02796 \\ -0.03594 \end{bmatrix}
$$

#### Gradient of $$ a_1 $$ with Respect to $$ z_1 $$ (Before ReLU)

Since $$ a_1 = \text{ReLU}(z_1) $$, the gradient of ReLU is $$ 1 $$ for positive values:

$$
\frac{\partial a_1}{\partial z_1} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

#### Gradient of the Loss with Respect to $$ z_1 $$

Now, the gradient of the loss with respect to $$ z_1 $$ is simply:

$$
\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \times \frac{\partial a_1}{\partial z_1} = \begin{bmatrix} -0.02796 \\ -0.03594 \end{bmatrix}
$$

#### Gradient of $$ z_1 $$ with Respect to $$ W_1 $$ (First Layer Weights)

Finally, the gradient of $$ z_1 $$ with respect to $$ W_1 $$ is the input $$ x $$:

$$
\frac{\partial z_1}{\partial W_1} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

#### Gradient of the Loss with Respect to $$ W_1 $$

Using the chain rule, we multiply:

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \times \frac{\partial z_1}{\partial W_1} = \begin{bmatrix} -0.02796 \\ -0.03594 \end{bmatrix} \times \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} -0.02796 & -0.07188 \\ -0.03594 & -0.08985 \end{bmatrix}
$$

### 4. Update the Weights

Now that we have the gradients, we update the weights $$ W_1 $$ and $$ W_2 $$ using gradient descent:

$$
W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}
$$

Where $$ \eta $$ is the learning rate.

## Conclusion

This example shows how a neural network makes predictions (forward pass) and adjusts its weights to improve (backward pass). We walked through:
- **Linear transformations** using matrix multiplication.
- **Activation functions** like ReLU and sigmoid.
- **Loss calculation** using binary cross-entropy.
- **Backpropagation** to compute the gradients and update the weights.
