---
layout: single
title:  "Generative AI 1.3: Probability and Statistics"
date:   2024-09-21
categories: [deep-learning]
tags: [probability, statistics, mathematics, foundations, generative-ai]
description: "Master probability and statistics fundamentals essential for understanding generative AI models like VAEs and GANs"
author: Dhiraj Salian
---

# Probability and Statistics for Generative AI

> Machine learning is essentially applied probability. Every prediction is a probability, every loss function is based on likelihood, and every model uncertainty is quantified statistically.

When a generative model creates an image or writes text, it's making probabilistic predictions. When we train it, we're maximizing likelihood. This guide covers the probability foundations you need.

## Random Variables

A **random variable** represents a quantity that can take different values according to a probability distribution.

**Discrete**: Countable values (coin flips, word choices)
**Continuous**: Unbounded range (pixel values, temperatures)

## Probability Distributions

A distribution assigns probabilities to possible outcomes.

### Discrete: Bernoulli

For a binary outcome (coin flip):
$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

### Continuous: Normal (Gaussian)

The most common distribution in ML:
$$P(X = x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

Where $\mu$ = mean, $\sigma^2$ = variance.

## Expectation and Variance

### Expectation (Mean)

The weighted average of possible values:
$$\mathbb{E}[X] = \sum_x P(X = x) \cdot x$$

**Example**: Fair coin flip
$$\mathbb{E}[X] = 1 \cdot 0.5 + 0 \cdot 0.5 = 0.5$$

### Variance

How spread out the distribution is:
$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$$

### Standard Deviation

Square root of variance—same units as data:
$$\text{Std}(X) = \sqrt{\text{Var}(X)}$$

## Conditional Probability and Bayes' Theorem

### Conditional Probability

Probability of A given B has occurred:
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Example**: 60% study math, 30% study both
$$P(\text{Physics}|\text{Math}) = \frac{0.3}{0.6} = 0.5$$

### Bayes' Theorem

Update beliefs with new evidence:
$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

This is the foundation of:
- **Naive Bayes** classifiers
- **Bayesian networks**
- **Bayesian inference** in neural networks

## Maximum Likelihood Estimation (MLE)

MLE finds the parameters that make the observed data most probable.

**Likelihood** of data given parameters $\theta$:
$$L(\theta) = P(D|\theta) = \prod_{i=1}^n P(x_i|\theta)$$

**Log-likelihood** (practical to compute):
$$\log L(\theta) = \sum_{i=1}^n \log P(x_i|\theta)$$

This is what neural networks optimize—maximizing the likelihood of training data.

## Implementation

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate normal distribution data
mean, std = 0, 1
data = np.random.normal(mean, std, 1000)

# Compute statistics
print(f"Mean: {np.mean(data):.4f}")      # ≈ 0
print(f"Variance: {np.var(data):.4f}")   # ≈ 1
print(f"Std Dev: {np.std(data):.4f}")    # ≈ 1

# Plot distribution
x = np.linspace(-4, 4, 1000)
plt.plot(x, norm.pdf(x, mean, std))
plt.title("Normal Distribution")
plt.grid(True)
plt.show()
```

## The Bottom Line

- **Random variables** model uncertainty in data
- **Distributions** describe how values are probabilistic
- **Expectation/variance** characterize distributions
- **Bayes' theorem** enables inference from evidence
- **MLE** is how we fit models to data

These concepts are the foundation of probabilistic ML and generative models.