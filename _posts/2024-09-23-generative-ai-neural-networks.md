---
layout: single
title:  "Generative AI 2: Neural Networks"
date:   2024-09-23
categories: artificial-intelligence
---

A **neural network** is a computational model inspired by the way biological neural networks work. It is composed of layers of neurons (nodes), which are connected by **weights**. Neural networks are used to approximate complex functions and learn patterns from data through training.

## 2.1 Components of a Neural Network

### 2.1.1 Neurons

Each neuron in a neural network receives inputs, processes them, and outputs a value. The neuron performs a linear transformation on the inputs, followed by a non-linear activation function.

### 2.1.2 Layers

- **Input Layer**: This is where the data enters the network.
- **Hidden Layers**: Intermediate layers between input and output layers. These perform transformations on the data.
- **Output Layer**: Produces the final result (e.g., predicted class label or value).

### 2.1.3 Weights and Biases

Each connection between neurons has a **weight** that determines how much influence the input has on the output. Neurons also have a **bias** term, which shifts the activation function.

### 2.1.4 Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn more complex patterns.

#### Common Activation Functions:
1. **ReLU (Rectified Linear Unit)**:
   \\[
   \\text{ReLU}(z) = \\max(0, z)
   \\]

2. **Sigmoid**:
   \\[
   \\sigma(z) = \\frac{1}{1 + e^{-z}}
   \\]

3. **Tanh**:
   \\[
   \\text{tanh}(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}
   \\]

## 2.2 Forward Propagation

In **forward propagation**, data moves through the layers of the network from the input to the output. Each layer computes a weighted sum of its inputs and applies an activation function to produce the output for the next layer.

### Mathematics of Forward Propagation:

For a layer with input \\(\\mathbf{x}\\), weights \\(\\mathbf{W}\\), and bias \\(\\mathbf{b}\\), the linear output is:

\\[
z = \\mathbf{W} \\cdot \\mathbf{x} + \\mathbf{b}
\\]

Then, the activation function \\(\\sigma(z)\\) is applied:

\\[
a = \\sigma(z)
\\]

This process repeats for each layer, where the output of one layer becomes the input to the next.

## 2.3 Loss Function

The **loss function** measures the difference between the predicted output and the true target. The goal of training a neural network is to minimize the loss.

### Common Loss Functions:
1. **Mean Squared Error (MSE)** for regression:
   \\[
   \\text{MSE} = \\frac{1}{n} \\sum_{i=1}^n (y_i - \\hat{y}_i)^2
   \\]
2. **Cross-Entropy Loss** for classification:
   \\[
   \\text{Loss} = -\\sum_{i=1}^n y_i \\log(\\hat{y}_i)
   \\]
Where:
- \\(y_i\\) is the true label, and \\(\\hat{y}_i\\) is the predicted probability.

## 2.4 Backpropagation

**Backpropagation** is the process of computing the gradients of the loss function with respect to the weights using the **chain rule**. The gradients are then used to update the weights to minimize the loss.

### Steps in Backpropagation:
1. **Compute the loss** based on the output from forward propagation.
2. **Compute the gradient of the loss** with respect to the output.
3. **Propagate the gradient backward** through the network, using the chain rule to compute gradients for the weights and biases.
4. **Update the weights** using an optimization algorithm like gradient descent.

## Implementing a Neural Network with PyTorch

Let’s implement a simple neural network using **PyTorch** to classify handwritten digits from the **MNIST dataset**.

### Installing torch
```bash
pip install torch torchvision torchaudio
```

### Python Code:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer (28x28 pixels = 784 inputs)
        self.fc2 = nn.Linear(128, 10)     # Output layer (10 digits)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = self.fc2(x)  # Output logits (no activation)
        return x

# 3. Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the neural network
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

# 5. Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
```

### Output:
```
Epoch 1/5, Loss: 0.2995
Epoch 2/5, Loss: 0.1396
Epoch 3/5, Loss: 0.1034
Epoch 4/5, Loss: 0.0846
Epoch 5/5, Loss: 0.0719
Test Accuracy: 97.68%
```
{: .no-copy}

## Summary:
- A neural network is composed of layers of neurons, each performing a linear transformation followed by a non-linear activation function.
- Forward propagation passes data through the network to generate predictions, and backpropagation is used to compute the gradients of the loss function.
- The network’s parameters (weights and biases) are updated using an optimizer like Adam or SGD.
- PyTorch provides tools to easily implement neural networks and handle both forward and backward propagation.
- For a detailed working of a simple neural network checkout - [Simple Neural Network in Python with Math behind it]({{ site.url }}{{ site.baseurl }}{% link _posts/2024-09-23-simplified-neural-network.md %})
