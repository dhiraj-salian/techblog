
import datetime
import os
import re

def slugify(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', '-', text)
    text = re.sub(r'[^\w\-]+', '', text)
    return text

def generate_blog_post(title, category, content, output_dir="techblog/_posts"):
    today = datetime.date.today()
    date_str = today.strftime("%Y-%m-%d")
    
    slug_title = slugify(title)
    filename = f"{date_str}-{slug_title}.md"
    filepath = os.path.join(output_dir, filename)

    markdown_content = f"""---
layout: single
title:  "{title}"
date:   {date_str}
categories: {category}
---

{content}
"""

    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(markdown_content)
    
    print(f"Generated blog post: {filepath}")
    return filepath

if __name__ == "__main__":
    # Example usage:
    # You would typically get these inputs from an AI model or a configured list
    # For now, we'll use a placeholder
    example_title = "Introduction to Python Programming"
    example_category = "python-programming"
    example_content = """
Welcome to the start of our Python Programming series! In this post, we'll cover the absolute basics of Python.

## What is Python?
Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and scientific computing.

## Setting up Python
You can download Python from its official website (python.org) or use a package manager. We recommend using a virtual environment for your projects.

```bash
python3 -m venv myenv
source myenv/bin/activate
```

## Your First Python Program
Let's write the classic "Hello, World!" program.

```python
print("Hello, World!")
```

## Basic Data Types
Python has several built-in data types:
- **Integers**: `age = 30`
- **Floats**: `price = 19.99`
- **Strings**: `name = "Alice"`
- **Booleans**: `is_active = True`

## Variables and Assignment
Variables are used to store data. You don't need to declare their type explicitly.

```python
x = 10
y = "OpenClaw"
z = 3.14
```

## Operators
Python supports various operators:
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Logical**: `and`, `or`, `not`

## Next Steps
In the next post, we will delve into control flow (if/else statements, loops) and functions.
"""
    
    generate_blog_post(example_title, example_category, example_content)

    example_title_2 = "Fundamentals of Machine Learning: An Overview"
    example_category_2 = "machine-learning"
    example_content_2 = """
This post introduces the fundamental concepts of Machine Learning, providing a high-level overview of what ML is, its main types, and common applications.

## What is Machine Learning?
Machine Learning (ML) is a subset of Artificial Intelligence (AI) that enables systems to learn from data, identify patterns, and make decisions with minimal human intervention. Instead of being explicitly programmed, ML models learn from data to improve their performance over time.

## Types of Machine Learning
There are three primary types of machine learning:

### 1. Supervised Learning
In supervised learning, the model is trained on a labeled dataset, meaning each data point has an associated output label. The goal is for the model to learn a mapping from inputs to outputs, allowing it to predict labels for new, unseen data.
- **Examples**: Classification (spam detection, image recognition), Regression (house price prediction, stock forecasting).

### 2. Unsupervised Learning
Unsupervised learning deals with unlabeled data. The model tries to find hidden patterns or structures in the data without any explicit guidance.
- **Examples**: Clustering (customer segmentation), Dimensionality Reduction (feature selection).

### 3. Reinforcement Learning
Reinforcement learning involves an agent learning to make decisions by performing actions in an environment to maximize a cumulative reward. The agent learns through trial and error.
- **Examples**: Game playing (AlphaGo), Robotics, Autonomous driving.

## Common Machine Learning Tasks
- **Classification**: Categorizing data into predefined classes.
- **Regression**: Predicting a continuous output value.
- **Clustering**: Grouping similar data points together.
- **Dimensionality Reduction**: Reducing the number of features in a dataset while retaining important information.

## The Machine Learning Workflow
A typical ML workflow includes:
1.  **Data Collection**: Gathering relevant data.
2.  **Data Preprocessing**: Cleaning, transforming, and preparing data.
3.  **Feature Engineering**: Creating new features from existing ones.
4.  **Model Selection**: Choosing an appropriate ML algorithm.
5.  **Model Training**: Training the model on the preprocessed data.
6.  **Model Evaluation**: Assessing the model's performance.
7.  **Hyperparameter Tuning**: Optimizing model parameters.
8.  **Deployment**: Integrating the model into an application.

## Applications of Machine Learning
ML is transforming various industries, with applications such as:
-   **Healthcare**: Disease diagnosis, drug discovery.
-   **Finance**: Fraud detection, algorithmic trading.
-   **E-commerce**: Recommendation systems, personalized shopping.
-   **Autonomous Vehicles**: Object detection, navigation.
-   **Natural Language Processing**: Sentiment analysis, language translation.

## Next Steps
In subsequent posts, we will dive deeper into each type of machine learning, explore specific algorithms, and implement them using Python libraries.
"""
    generate_blog_post(example_title_2, example_category_2, example_content_2)
