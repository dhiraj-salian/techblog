---
layout: single
title:  "Generative AI 1.1: Linear Algebra"
date:   2024-09-19
categories: [deep-learning]
tags: [linear-algebra, mathematics, foundations, generative-ai]
description: "Master the linear algebra concepts that power every neural network—from vectors to matrix operations"
author: Dhiraj Salian
---

# Linear Algebra for Generative AI

> Every neural network, at its core, is a series of matrix multiplications. Understanding linear algebra isn't optional for ML—it's the foundation everything builds on.

If you've ever wondered how a neural network processes data, transforms inputs to outputs, or learns from examples—the answer is linear algebra.

This guide covers the linear algebra essentials you'll need for generative AI.

## Vectors: The Building Blocks

A **vector** is an ordered list of numbers. In ML, vectors represent:
- Input data (e.g., an image as a list of pixel values)
- Model weights (learned parameters)
- Embeddings (dense representations of text, images, etc.)

$$\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$$

### Essential Operations

**Vector Addition**: Element-wise addition
$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}$$

**Scalar Multiplication**: Scaling each element
$$c \mathbf{v} = \begin{bmatrix} c v_1 \\ c v_2 \end{bmatrix}$$

## Matrices: Transformations

A **matrix** is a 2D array of numbers. Matrices represent **linear transformations**—the core operation in neural networks.

$$W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

### Matrix-Vector Multiplication

This is where neural networks actually "compute." An input vector passes through a weight matrix to produce an output:

$$\mathbf{y} = W \mathbf{x}$$

**Example**:
$$W = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$$

$$W \mathbf{x} = \begin{bmatrix} 1 \cdot 3 + 2 \cdot 4 \\ 3 \cdot 3 + 4 \cdot 4 \end{bmatrix} = \begin{bmatrix} 11 \\ 25 \end{bmatrix}$$

### Matrix-Matrix Multiplication

Neural network layers can be composed—stacking multiple transformations:

$$W_3 = W_1 W_2$$

## The Dot Product: Weighted Summations

The **dot product** computes weighted sums—a fundamental operation in every neuron:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i$$

**Example**:
For $\mathbf{a} = [1, 2, 3]$ and $\mathbf{b} = [4, 5, 6]$:
$$\mathbf{a} \cdot \mathbf{b} = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 32$$

## Norms and Distance

### L2 Norm (Magnitude)

The L2 norm measures a vector's length—used in regularization and loss functions:

$$||\mathbf{v}||_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

### Euclidean Distance

Distance between vectors—found in similarity measures and clustering:

$$d(\mathbf{a}, \mathbf{b}) = ||\mathbf{a} - \mathbf{b}||_2$$

**Example**: Distance between $\mathbf{a} = [1, 2]$ and $\mathbf{b} = [3, 4]$:
$$d(\mathbf{a}, \mathbf{b}) = \sqrt{(1-3)^2 + (2-4)^2} = \sqrt{8} \approx 2.83$$

## Practical Implementation

```python
import numpy as np

# Vectors
a = np.array([1, 2])
b = np.array([3, 4])

# Addition
print("a + b:", a + b)  # [4 6]

# Dot product
print("a · b:", np.dot(a, b))  # 11

# Matrix
W = np.array([[3, 4], [5, 6]])

# Matrix-vector multiplication
print("W @ a:", np.dot(W, a))  # [11 17]

# L2 norm
print("||a||₂:", np.linalg.norm(a))  # 2.236

# Distance
print("distance(a,b):", np.linalg.norm(a - b))  # 2.828
```

## The Bottom Line

- **Vectors** represent data points and model parameters
- **Matrices** perform transformations (the core of neural networks)
- **Dot products** compute weighted sums in neurons
- **Norms** measure magnitudes (crucial for loss functions and regularization)

This is the language of neural networks. Master it, and you'll understand not just how ML works, but why it works that way.