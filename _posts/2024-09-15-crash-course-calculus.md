---
layout: single
title:  "Crash Course: Calculus"
date:   2024-09-15
categories: mathematics
---

Calculus is a branch of mathematics focused on **change** (differentiation) and **accumulation** (integration). It is divided into **differential calculus** and **integral calculus**.

## 1. Differential Calculus

**Goal**: Understand how things change.

**Key Concept**: **Derivative**
- A derivative measures the rate of change of a function with respect to one of its variables.

### Notation:
If \\( y = f(x) \\), the derivative of \\( y \\) with respect to \\( x \\) is written as:
\\[
\frac{dy}{dx} \quad \text{or} \quad f'(x)
\\]
This gives the slope of the function at any point \\( x \\).

### Basic Rules:
- **Constant Rule**: The derivative of a constant is zero.
  \\[
  \frac{d}{dx}(c) = 0
  \\]
- **Power Rule**: The derivative of \\( x^n \\) is \\( nx^{n-1} \\).
  \\[
  \frac{d}{dx}(x^n) = nx^{n-1}
  \\]
- **Sum Rule**: The derivative of a sum is the sum of the derivatives.
  \\[
  \frac{d}{dx}(f(x) + g(x)) = f'(x) + g'(x)
  \\]
- **Product Rule**: The derivative of a product is:
  \\[
  \frac{d}{dx}(f(x)g(x)) = f'(x)g(x) + f(x)g'(x)
  \\]
- **Quotient Rule**: The derivative of a quotient is:
  \\[
  \frac{d}{dx}\left(\frac{f(x)}{g(x)}\right) = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}
  \\]
- **Chain Rule**: The derivative of a composite function is:
  \\[
  \frac{d}{dx}(f(g(x))) = f'(g(x))g'(x)
  \\]

### Geometrically:
The derivative gives the slope of the tangent line to the curve of \\( f(x) \\) at a specific point.

## 2. Integral Calculus

**Goal**: Understand how to accumulate or total quantities over a range.

**Key Concept**: **Integral**
- An integral represents the area under a curve. It can be thought of as the reverse operation of differentiation.

### Notation:
The indefinite integral (anti-derivative) of \\( f(x) \\) with respect to \\( x \\) is written as:
\\[
\int f(x) \, dx
\\]
The definite integral of \\( f(x) \\) from \\( a \\) to \\( b \\) is:
\\[
\int_{a}^{b} f(x) \, dx
\\]
This calculates the area under the curve from \\( x = a \\) to \\( x = b \\).

### Basic Rules:
- **Constant Rule**: The integral of a constant is:
  \\[
  \int c \, dx = cx + C
  \\]
  where \\( C \\) is the constant of integration.
- **Power Rule**: The integral of \\( x^n \\) is:
  \\[
  \int x^n \, dx = \frac{x^{n+1}}{n+1} + C \quad \text{(for \\( n \neq -1 \\))}
  \\]
- **Sum Rule**: The integral of a sum is the sum of the integrals.
  \\[
  \int (f(x) + g(x)) \, dx = \int f(x) \, dx + \int g(x) \, dx
  \\]

### Fundamental Theorem of Calculus:
\\[
\frac{d}{dx} \left( \int_{a}^{x} f(t) \, dt \right) = f(x)
\\]
It connects differentiation and integration, showing that differentiation reverses integration and vice versa.

## 3. Applications of Calculus

- **Differential Calculus** is used to:
  - Find instantaneous rates of change (e.g., velocity, acceleration).
  - Maximize or minimize functions (e.g., optimizing profits or efficiency).

- **Integral Calculus** is used to:
  - Calculate areas under curves, volumes of solids.
  - Accumulate quantities (e.g., total distance, mass, or charge over time or space).

## Summary:
- **Derivative**: Measures how a function changes.
- **Integral**: Measures how much a function accumulates.
- **Key Rules**: Power rule, product rule, chain rule (for derivatives); power rule, sum rule (for integrals).
