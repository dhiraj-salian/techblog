---
layout: single
title:  "Generative AI 1.1: Linear Algebra"
date:   2024-09-19
categories: artificial-intelligence
---

Linear algebra is the foundation of machine learning and neural networks. In particular, **vectors** and **matrices** are used to represent and manipulate data, weights, and transformations within these models.

## 1.1.1 Vectors

A **vector** is an ordered list of numbers. Vectors represent various elements in machine learning, such as input data, weights, or model parameters.

### Vector Example:
\\[
\\mathbf{v} = \\begin{bmatrix} 3 \\\\ 4 \\end{bmatrix}
\\]

### Operations on Vectors:

1. **Vector Addition**:

    You can add two vectors element-wise.
    \\[
    \\mathbf{a} + \\mathbf{b} = \\begin{bmatrix} a_1 \\\\ a_2 \\end{bmatrix} + \\begin{bmatrix} b_1 \\\\ b_2 \\end{bmatrix} = \\begin{bmatrix} a_1 + b_1 \\\\ a_2 + b_2 \\end{bmatrix}
    \\]

2. **Scalar Multiplication**:

    Multiply a vector by a scalar (single number) by multiplying each element by the scalar.
    \\[
    c \\mathbf{v} = c \\begin{bmatrix} v_1 \\\\ v_2 \\end{bmatrix} = \\begin{bmatrix} c v_1 \\\\ c v_2 \\end{bmatrix}
    \\]

## 1.1.2 Matrices

A **matrix** is a rectangular array of numbers that can represent transformations, such as those used in neural networks to map inputs to outputs.

### Matrix Example:
\\[
W = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}
\\]

### Matrix Operations:

1. **Matrix-Vector Multiplication**:
    
    This operation is used in neural networks to transform input vectors using weight matrices.
    \\[
    \\mathbf{y} = W \\mathbf{x}
    \\]
    Where:
    - \\(W\\) is the matrix,
    - \\(\\mathbf{x}\\) is the input vector,
    - \\(\\mathbf{y}\\) is the output vector.

    **Example**:
    \\[
    W = \\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}, \\quad \\mathbf{x} = \\begin{bmatrix} 3 \\\\ 4 \\end{bmatrix}
    \\]
    \\[
    W \\mathbf{x} = \\begin{bmatrix} 1 \\cdot 3 + 2 \\cdot 4 \\\\ 3 \\cdot 3 + 4 \\cdot 4 \\end{bmatrix} = \\begin{bmatrix} 11 \\\\ 25 \\end{bmatrix}
    \\]

2. **Matrix-Matrix Multiplication**:
    
    Multiply two matrices by performing dot products between rows of the first matrix and columns of the second.
    \\[
    W_3 = W_1 W_2
    \\]

## 1.1.3 Dot Product

The **dot product** between two vectors produces a scalar value and is often used in neural networks for computing weighted sums.

### Dot Product Formula:
\\[
\\mathbf{a} \\cdot \\mathbf{b} = a_1 b_1 + a_2 b_2 + \\cdots + a_n b_n = \\sum_{i=1}^n a_i b_i
\\]

### Dot Product Example:
For \\(\\mathbf{a} = [1, 2, 3]\\) and \\(\\mathbf{b} = [4, 5, 6]\\):
\\[
\\mathbf{a} \\cdot \\mathbf{b} = 1 \\cdot 4 + 2 \\cdot 5 + 3 \\cdot 6 = 4 + 10 + 18 = 32
\\]

## 1.1.4 Norms and Distance

The **norm** of a vector measures its magnitude (or length), while the **distance** between two vectors is the Euclidean distance between them.

### L2 Norm (Euclidean Norm):
The **L2 norm** measures the magnitude of a vector:
\\[
||\\mathbf{v}||_2 = \\sqrt{v_1^2 + v_2^2 + \\cdots + v_n^2}
\\]

### Distance Between Vectors:
The **distance** between two vectors \\(\\mathbf{a}\\) and \\(\\mathbf{b}\\) is:
\\[
d(\\mathbf{a}, \\mathbf{b}) = ||\\mathbf{a} - \\mathbf{b}||_2
\\]

### Example:
If \\(\\mathbf{a} = [1, 2]\\) and \\(\\mathbf{b} = [3, 4]\\), the distance is:
\\[
d(\\mathbf{a}, \\mathbf{b}) = \\sqrt{(1 - 3)^2 + (2 - 4)^2} = \\sqrt{4 + 4} = \\sqrt{8} \\approx 2.83
\\]

## Working with Vectors and Matrices in Python

We will be using python library `numpy` for performing the above linear algebra operations.

### Install numpy:
```bash
pip install numpy
```

### Python Code:
```python
import numpy as np

# Define vectors
a = np.array([1, 2])
b = np.array([3, 4])

# Vector addition
c = a + b
print("Vector addition:", c)

# Dot product
dot_product = np.dot(a, b)
print("Dot product:", dot_product)

# Define a matrix
W = np.array([[3, 4], [5, 6]])

# Matrix-vector multiplication
result = np.dot(W, a)
print("Matrix-vector multiplication:", result)

# Matrix-matrix multiplication
W2 = np.array([[1, 0], [0, 1]])
matrix_product = np.dot(W, W2)
print("Matrix-matrix multiplication:", matrix_product)

# Compute the L2 norm (Euclidean norm) of a vector
norm_a = np.linalg.norm(a)
print("L2 norm of vector a:", norm_a)

# Compute the distance between two vectors
distance = np.linalg.norm(a - b)
print("Distance between a and b:", distance)
```

### Expected Output:
```
Vector addition: [4 6]
Dot product: 11
Matrix-vector multiplication: [11 17]
Matrix-matrix multiplication: [[3 4]
 [5 6]]
L2 norm of vector a: 2.23606797749979
Distance between a and b: 2.8284271247461903
```
{: .no-copy}

## Summary:
- Vectors are used to represent input data, model parameters, and outputs.
- Matrices are used to transform input vectors (through matrix-vector multiplication) in neural networks.
- The dot product computes weighted sums in neurons.
- The L2 norm measures the magnitude of a vector, and the distance measures the Euclidean distance between vectors.