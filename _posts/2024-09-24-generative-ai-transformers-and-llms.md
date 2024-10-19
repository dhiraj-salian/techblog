---
layout: single
title:  "Generative AI 3: Transformer Networks and Large Language Models (LLMs)"
date:   2024-09-24
categories: artificial-intelligence
---

**Transformers** are the foundation of modern **Large Language Models (LLMs)** such as **GPT-3**, **BERT**, and **T5**. In this step, we will explore the transformer architecture, its components, and how it enables LLMs to process and generate human-like text. We will also learn about **tokenizers**, which are critical for preparing text data for transformer models.

### **3.1 Transformer Architecture**

The transformer is a deep learning architecture designed to process sequences of data in parallel using a mechanism called **self-attention**. It has become the dominant model for NLP tasks due to its ability to handle long-range dependencies and scale to large datasets.

#### **Key Components of Transformers**:

1. **Self-Attention**:
   - The **self-attention** mechanism allows the model to focus on different parts of the input sequence to understand relationships between words, regardless of their position in the sequence.
   - It computes a weighted sum of all input tokens, where the weights (or "attention scores") represent the importance of each token relative to others.

   **Formula**:
   $$
   \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
   $$
   Where:
   - $Q$ (Query), $K$ (Key), and $V$ (Value) are matrices representing the input tokens.
   - $d_k$ is the dimension of the key vectors.

2. **Multi-Head Attention**:
   - Instead of a single attention function, the transformer uses multiple attention heads to focus on different parts of the input. Each attention head operates independently, and their outputs are combined.

3. **Position-Wise Feed-Forward Networks**:
   - After attention, each token passes through a feed-forward network. This network is applied independently to each token, which helps in transforming the attended input.

4. **Positional Encoding**:
   - Transformers process input in parallel, so they need **positional encodings** to understand the order of tokens in a sequence. These encodings are added to the input embeddings.

5. **Encoder and Decoder**:
   - **Encoder**: The encoder reads the input sequence and processes it through self-attention and feed-forward layers.
   - **Decoder**: The decoder generates the output sequence by attending to both the encoder's output and the previously generated tokens.

### **3.2 Tokenization**

Before feeding text into a transformer model, it needs to be converted into tokens (numerical representations). **Tokenization** is the process of splitting text into smaller units like words, subwords, or characters.

#### **Types of Tokenizers**:

1. **Word Tokenizers**:
   - These split text into individual words.
   - Example: `"I love cats"` → `["I", "love", "cats"]`

2. **Subword Tokenizers**:
   - These split rare words into smaller, meaningful subword units. Common techniques include:
     - **Byte Pair Encoding (BPE)** (used in GPT-2, GPT-3).
     - **WordPiece** (used in BERT).
   - Example: `"unhappiness"` → `["un", "##happiness"]`

3. **Character Tokenizers**:
   - These split text into individual characters.
   - Example: `"hello"` → `["h", "e", "l", "l", "o"]`

#### **Why Tokenization is Important**:
- Tokenization is crucial for transforming text into a format that LLMs can process. Each token is mapped to an embedding, which serves as the model's input.

#### **Practical Tokenization with Hugging Face**:
We’ll use the **Hugging Face Transformers** library to load pre-trained tokenizers and process text for transformer models like GPT-2 and BERT.

### **3.3 Fine-Tuning Large Language Models (LLMs)**

#### **Pre-trained LLMs**:
LLMs like **GPT**, **BERT**, and **T5** are pre-trained on massive text datasets using unsupervised learning objectives. These models can be fine-tuned on specific downstream tasks such as text classification, text generation, or summarization.

- **GPT (Generative Pre-trained Transformer)**: Uses **causal language modeling** to predict the next word in a sequence.
- **BERT (Bidirectional Encoder Representations from Transformers)**: Uses **masked language modeling** to predict masked words in a sentence.
- **T5 (Text-to-Text Transfer Transformer)**: Treats all NLP tasks as text-to-text problems, making it highly flexible.

### **Tokenization and Fine-Tuning with Hugging Face**

#### **Tokenization Example**

Let’s tokenize some text using a pre-trained **BERT** tokenizer.

```python
from transformers import BertTokenizer

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example sentence
sentence = "I love learning about transformers."

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)
print("Tokens:", tokens)

# Convert tokens to input IDs
input_ids = tokenizer.encode(sentence, add_special_tokens=True)
print("Input IDs:", input_ids)
```

#### **Output**:

```
Tokens: ['i', 'love', 'learning', 'about', 'transformers', '.']
Input IDs: [101, 1045, 2293, 4083, 2055, 19081, 1012, 102]
```

Here, special tokens [101] (start token) and [102] (end token) are added automatically.

#### **Fine-Tuning Example**

Next, let's fine-tune a pre-trained BERT model on a text classification task (e.g., sentiment analysis).

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load a dataset (for example, IMDb sentiment dataset)
dataset = load_dataset('imdb')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    evaluation_strategy="epoch",     
    save_steps=10_000,               
    save_total_limit=2,              
    logging_dir='./logs',            
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=tokenized_datasets['train'],         
    eval_dataset=tokenized_datasets['test'],          
)

# Train the model
trainer.train()
```

#### **Output**:

```
***** Running training *****
  Num examples = 25000
  Num Epochs = 3
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 1
  Total optimization steps = 4688

Epoch    Training Loss   Validation Loss   Accuracy
---------------------------------------------------
  1/3    0.4132          0.3741            0.8327
  2/3    0.2476          0.3238            0.8654
  3/3    0.1567          0.3121            0.8750

***** Running Evaluation *****
  Num examples = 25000
  Batch size = 64
Final Accuracy on test set: 87.50%
```

### **Summary of Transformers and LLMs**:
- Transformers are the backbone of modern Large Language Models (LLMs) like GPT, BERT, and T5. They use self-attention mechanisms to process and generate text efficiently.
- Tokenizers are essential for converting text into numerical tokens that LLMs can understand. Pre-trained tokenizers like those from Hugging Face handle this process automatically.
- Fine-tuning allows you to adapt pre-trained LLMs to specific tasks (e.g., text classification, generation) using your own dataset.
