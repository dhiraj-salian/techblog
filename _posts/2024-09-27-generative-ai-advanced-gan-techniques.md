---
layout: single
title:  "Generative AI 6: Advanced GAN Techniques"
date:   2024-09-27
categories: artificial-intelligence
---

In this step, we will explore advanced variants of **GANs** that address some limitations of the basic GAN architecture, such as unstable training and lack of control over generated samples. These techniques make GANs more powerful and adaptable for real-world applications, especially in generating high-quality images, videos, and other data.

### **6.1 Conditional GANs (cGANs)**

#### **What are Conditional GANs (cGANs)?**

**Conditional GANs (cGANs)** extend the basic GAN architecture by conditioning both the **generator** and **discriminator** on additional information. For example, you can condition the GAN on **class labels**, which allows you to generate specific types of data (e.g., images of cats, dogs, or cars).

#### **How it Works**:
- **Generator**: The generator takes both the noise vector \( z \) and a class label \( y \) as input, and it generates data that corresponds to the given class.
- **Discriminator**: The discriminator takes both the real or generated data and the class label \( y \) as input, and it tries to determine whether the data is real or fake given the class label.

#### **Objective**:
The generator aims to generate data that not only fools the discriminator but also matches the desired class label. The discriminator tries to identify whether the data is real or fake, taking into account the class label.

#### **cGAN Loss Functions**:
1. **Generator Loss**:
   \[
   \mathcal{L}_G = -\log(D(G(z|y)|y))
   \]
   Here, \( G(z|y) \) is the generated data conditioned on label \( y \), and \( D(G(z|y)|y) \) is the discriminator’s output for the generated data conditioned on label \( y \).

2. **Discriminator Loss**:
   \[
   \mathcal{L}_D = -\left[ \log(D(x|y)) + \log(1 - D(G(z|y)|y)) \right]
   \]
   Here, \( D(x|y) \) is the discriminator’s output for real data \( x \) conditioned on label \( y \), and \( G(z|y) \) is the fake data.

#### **Implementing a cGAN with PyTorch**

Here’s how you can implement a **Conditional GAN** in PyTorch to generate specific classes of images (e.g., digits from the MNIST dataset).

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Generator for cGAN
class cGANGenerator(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super(cGANGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Concatenate noise and label embeddings
        gen_input = torch.cat((self.label_embedding(labels), noise), dim=1)
        img = self.model(gen_input)
        return img.view(img.size(0), 1, 28, 28)

# Discriminator for cGAN
class cGANDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(cGANDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(28 * 28 + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Concatenate image and label embeddings
        d_input = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), dim=1)
        validity = self.model(d_input)
        return validity

# Training loop for cGAN (similar to basic GAN, but with label conditioning)
```

#### **Applications**:
- **Image generation**: Generate specific types of images (e.g., digits, animals, objects) by conditioning on class labels.
- **Text-to-image**: You can generate images from text descriptions by conditioning the GAN on text features.

### **6.2 Deep Convolutional GANs (DCGANs)**

#### **What are DCGANs?**
Deep Convolutional GANs (DCGANs) are a variant of GANs that use convolutional layers in the generator and discriminator to improve the quality of generated images. DCGANs are particularly effective for generating high-quality images.

#### **Architecture**:

**Generator**: Uses transpose convolutional layers (also known as deconvolutional layers) to upsample the input noise and generate realistic images.
**Discriminator**: Uses standard convolutional layers to downsample the input image and classify whether it’s real or fake.

#### **Key Principles of DCGANs**:

1. **Convolutions instead of Fully Connected Layers**: Replaces fully connected layers with convolutional layers for better spatial understanding.
2. **Batch Normalization**: Uses batch normalization to stabilize training.
3. **ReLU Activation in Generator**: The generator uses ReLU activations except for the output layer, which uses Tanh.
4. **LeakyReLU in Discriminator**: The discriminator uses LeakyReLU activation for better gradient flow.

### Implementing a DCGAN with PyTorch

Here’s a sample implementation of a DCGAN to generate images from the CIFAR-10 dataset.