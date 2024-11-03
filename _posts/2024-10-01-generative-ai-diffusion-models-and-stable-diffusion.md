---
layout: single
title:  "Generative AI 8: Diffusion Models and Stable Diffusion"
date:   2024-10-01
categories: artificial-intelligence
---

**Diffusion models** are a class of generative models that create data by learning the reverse process of data corruption. In the case of **text-to-image** generation models like **Stable Diffusion**, the model starts with random noise and iteratively refines it to generate an image based on a given text prompt. These models have gained popularity due to their ability to generate high-quality, diverse, and realistic images from text descriptions.

## 8.1 What are Diffusion Models?

### Diffusion Process:
- **Forward Process (Noise Addition)**: In diffusion models, data (e.g., an image) is progressively corrupted by adding noise over multiple steps until it becomes pure noise.
- **Reverse Process (Noise Removal)**: The model learns to reverse this process by gradually removing noise from a noisy input to generate a realistic image.

### Mathematical Overview:
1. **Forward Process**:
   The data $$ \mathbf{x}_0 $$ is corrupted by adding Gaussian noise over $$ T $$ timesteps.

   $$
   q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t} \mathbf{x}_{t-1}, (1 - \alpha_t) \mathbf{I})
   $$

   Here, $$ \mathbf{x}_t $$ represents the noisy image at step $$ t $$, and $$ \alpha_t $$ controls the amount of noise added.
2. Reverse Process:
   The model learns to predict the denoised image at each timestep.

   $$
   p(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))
   $$
   
   The reverse process is modeled by a neural network that predicts the clean image step-by-step.

### Objective:
The model's goal is to generate realistic images by learning how to reverse the noisy forward process. The final generated image is obtained by iteratively denoising starting from pure noise.

## 8.2 Stable Diffusion

**Stable Diffusion** is a state-of-the-art **text-to-image** model that uses diffusion processes to generate high-quality images from text prompts. Unlike traditional GAN-based image generation models, Stable Diffusion relies on diffusion models for sampling.

### How Stable Diffusion Works:
1. **Text Input**: The user provides a textual description (e.g., "A cat sitting on a beach during sunset").
2. **Text Encoding**: The text is encoded using a pre-trained language model (e.g., **CLIP**) to capture the semantic meaning of the text.
3. **Noise Initialization**: The model starts with a noisy image and uses the encoded text as guidance to refine the image through a series of denoising steps.
4. **Denoising Steps**: At each step, the model predicts how the noise should be reduced to generate an image that aligns with the text description.
5. **Final Image**: After the final denoising step, the model outputs a high-quality image that matches the text prompt.

## Generating Images with Stable Diffusion

You can generate images using **Stable Diffusion** by leveraging the **Hugging Face Diffusers** library. Here’s a step-by-step example to generate an image from a text prompt.

### Python Code:

```python
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

# Define the text prompt
prompt = "A futuristic cityscape at sunset with flying cars"

# Generate the image
image = pipe(prompt).images[0]

# Display the generated image
plt.imshow(image)
plt.axis('off')
plt.show()
```

### Output:
![Stable Diffusion Output 1]({{ 'assets/images/sd_output_1.png' | relative_url }})

### Explanation:
- **StableDiffusionPipeline**: This is the pipeline that handles text-to-image generation using Stable Diffusion.
- **Prompt**: You define a text prompt that describes the image you want to generate (e.g., "A futuristic cityscape at sunset with flying cars").
- **Image Generation** : The model generates an image based on the input text.

## 8.4 Latent Diffusion Models (LDMs)
Stable Diffusion is based on Latent Diffusion Models (LDMs), which operate in a latent space rather than directly on pixel data. This makes the model more efficient in terms of memory and computation.

### Key Features of LDMs:
- **Efficiency**: LDMs compress images into a lower-dimensional latent space, making the diffusion process faster and less memory-intensive.
- **High-Quality Generation**: Despite working in latent space, LDMs can generate high-resolution images by learning the key features of the image in compressed form.

## 8.5 Customizing Stable Diffusion
You can fine-tune Stable Diffusion or adjust various parameters to customize the generated images:

### Control Parameters:
- **Guidance Scale**: Controls how strongly the image generation is guided by the text prompt. A higher guidance scale leads to images that more closely match the text prompt.
- **Number of Inference Steps**: Determines the number of denoising steps. More steps generally produce higher-quality images but take longer.

### Python Code:
```python
# Customizing Stable Diffusion parameters
guidance_scale = 7.5  # Strength of text guidance
num_inference_steps = 50  # Number of denoising steps

# Generate an image with custom parameters
image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]

# Display the generated image
plt.imshow(image)
plt.axis('off')
plt.show()
```

### Output:
![Stable Diffusion Output 2]({{ 'assets/images/sd_output_2.png' | relative_url }})


## 8.6 Applications of Stable Diffusion
1. **Art Generation**:
Artists and designers use Stable Diffusion to generate creative and unique artwork from textual descriptions.
2. **Image Super-Resolution**:
Diffusion models can be applied to image upscaling tasks, where low-resolution images are enhanced to high-resolution versions.
3. **Medical Imaging**:
In healthcare, diffusion models are being explored for generating synthetic medical images for research and training purposes.
4. **Gaming and Virtual Worlds**:
Game developers use text-to-image generation to quickly create game assets such as characters, landscapes, and objects.

## 8.7 Key Advantages of Diffusion Models
1. **High Quality and Diversity**: Diffusion models can generate highly realistic and diverse images, even for complex text prompts.
2. **Fine Control**: Users have control over the image generation process through parameters like guidance scale and number of inference steps.
3. **Reduced Mode Collapse**: Unlike GANs, which can suffer from mode collapse (generating limited types of outputs), diffusion models tend to generate a wider variety of outputs.

## Summary:
- Diffusion models generate data by learning to reverse a noise-adding process. They start with random noise and iteratively refine it to produce realistic data.
- Stable Diffusion is a state-of-the-art text-to-image model that generates high-quality images based on text prompts. It uses latent diffusion models for computational efficiency.
- Hugging Face Diffusers provides a convenient way to use pre-trained diffusion models for text-to-image generation tasks.
