---
layout: single
title:  "Machine Learning Fundamentals: SVM, Decision Trees, and Random Forests"
date:   2026-03-24
categories: [machine-learning]
tags: [machine-learning, SVM, decision-trees, random-forests, classification, ensemble-learning]
---

Welcome to today's machine learning deep dive! In 2026, these classic algorithms remain foundational for classification and regression tasks. Let's explore each one with practical implementation using scikit-learn.

## Decision Trees: Intuitive Machine Learning

Decision Trees mimic human decision-making with flowchart-like structures. They're versatile classifiers and regressors that break complex decisions into simple questions.

### Key Concepts

- **Gini Impurity:** Measures the probability of incorrect classification
- **Entropy:** Information gain-based splitting criterion
- **Pruning:** Prevents overfitting by limiting tree depth

### Implementation with scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
dt_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = dt_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

> **Pro Tip:** Visualize your decision tree using `tree.plot_tree()` to understand feature importance and decision paths!

---

## Random Forests: Ensemble Learning Power

Random Forests address overfitting in Decision Trees by combining multiple trees through ensemble learning (bagging - Bootstrap Aggregating).

### How It Works

1. **Bootstrap Sampling:** Each tree is trained on a random subset of data
2. **Feature Randomness:** At each split, only a random subset of features is considered
3. **Voting/Averaging:** Final prediction aggregates all tree predictions

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=15,            # Limit tree depth
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=2,      # Minimum samples in leaf
    max_features='sqrt',     # Features to consider at each split
    bootstrap=True,          # Use bootstrap sampling
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Feature Importance

Random Forests provide built-in feature importance:

```python
importances = rf_model.feature_importances_
for feature, importance in sorted(zip(feature_names, importances), 
                                   key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance:.4f}")
```

---

## Support Vector Machines: Maximum Margin Classification

SVMs are powerful supervised learning models that find the optimal hyperplane separating different classes while maximizing the margin between them.

### Key Concepts

- **Hyperplane:** Decision boundary in n-dimensional space
- **Support Vectors:** Data points closest to the hyperplane
- **Kernel Trick:** Transforms data to handle non-linear boundaries

### Kernel Functions

| Kernel | Use Case |
|--------|----------|
| Linear | Linearly separable data |
| RBF (Radial Basis Function) | Non-linear boundaries |
| Polynomial | Complex decision surfaces |

### Implementation with Different Kernels

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Important: SVMs require feature scaling!
# Create pipeline with StandardScaler
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

# Linear kernel for linearly separable data
svm_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear', C=1.0))
])

# Polynomial kernel
svm_poly = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='poly', degree=3, C=1.0))
])

# Train all models
svm_pipeline.fit(X_train, y_train)
y_pred_svm = svm_pipeline.predict(X_test)
print(f"SVM (RBF) Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1],
    'svm__kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
```

---

## Comparing All Three Models

```python
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Train all models
models = {
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'SVM (RBF)': svm_pipeline
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(10, 6))
names = list(results.keys())
scores = list(results.values())
plt.bar(names, scores, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylabel('Accuracy')
plt.title('Model Comparison: Decision Tree vs Random Forest vs SVM')
plt.ylim(0, 1)
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
```

---

## When to Use Which?

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Decision Tree** | Interpretability needs | Easy to visualize, handles both numerical and categorical | Prone to overfitting |
| **Random Forest** | Most classification tasks | Robust to overfitting, handles missing values | Less interpretable, slower than single tree |
| **SVM** | High-dimensional data, clear margins | Effective in high dimensions, memory efficient | Doesn't scale well to large datasets, requires scaling |

---

## Resources for Further Learning

- [scikit-learn Decision Trees Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Random Forests in Machine Learning - DigitalOcean](https://www.digitalocean.com/community/tutorials/random-forest-in-machine-learning)
- [Understanding SVM - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)
- [MIT OpenCourseWare: Machine Learning](https://ocw.mit.edu/courses/6-867-machine-learning-fall-2006/)

---

## Conclusion

These three algorithms form the backbone of classical machine learning. While deep learning has taken center stage for complex tasks like NLP and image recognition, SVMs, Decision Trees, and Random Forests remain excellent choices for:

- **Quick prototyping** and baseline models
- **Interpretability requirements** (especially Decision Trees)
- **Tabular data** where they often match or outperform neural networks
- **Resource-constrained environments** where simpler models are preferred

Practice implementing these algorithms on datasets from [Kaggle](https://www.kaggle.com/datasets) to solidify your understanding!

*What's your experience with these algorithms? Drop your questions in the comments below!*