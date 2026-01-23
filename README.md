# Perceiver Architecture Study: Minimal PyTorch Re-Implementation (CIFAR-10)

A compact, educational PyTorch reproduction of the **Perceiver** architecture introduced by Jaegle et al. (DeepMind, ICML 2021). This project implements the core ideas **Fourier positional encoding**, a **cross-attention bottleneck** into a learned **latent array**, and a **latent Transformer tower** and trains the resulting model on **CIFAR-10** to demonstrate the architecture end-to-end.

**Original paper:** *Perceiver: General Perception with Iterative Attention*  
https://arxiv.org/abs/2103.03206

**Authors:** Chetana Chakrapani; Ayushman Mishra

**GitHub:**  https://github.com/chetana0070; https://github.com/aymisxx

---

## Table of Contents
- [Motivation](#motivation)
- [Key Concepts](#key-concepts)
  - [1) Input tokenization (convolution-free)](#1-input-tokenization-convolution-free)
  - [2) Fourier positional encoding (2D)](#2-fourier-positional-encoding-2d)
  - [3) Cross-attention bottleneck (input → latents)](#3-cross-attention-bottleneck-input--latents)
  - [4) Latent Transformer tower (latents ↔ latents)](#4-latent-transformer-tower-latents--latents)
  - [5) Classification head](#5-classification-head)
- [Architecture Overview](#architecture-overview)
- [Mathematical Formulation](#mathematical-formulation)
  - [Attention](#attention)
  - [Fourier features](#fourier-features)
  - [Complexity](#complexity)
- [Implementation Notes](#implementation-notes)
- [Training Setup](#training-setup)
- [Results](#results)
- [Discussion](#discussion)
- [Reproducibility](#reproducibility)
- [Limitations & Future Work](#limitations--future-work)
- [Citation](#citation)

---

## Motivation
Standard Transformer self-attention scales quadratically with the number of input tokens **N**:

$$O(N^2)$$

For high-dimensional modalities (images, audio, video, point clouds), **N** can become extremely large, making vanilla Transformers expensive.

The Perceiver addresses this by introducing a learned **latent array** of size **M** (typically $M \ll N$) and using **cross-attention** to compress the input into latents. Subsequent computation happens only within latents, allowing compute to scale with $N \cdot M$ and $M^2$, rather than $N^2$.

---

## Key Concepts

### 1) Input tokenization (convolution-free)
This reproduction avoids convolutions entirely:
- CIFAR-10 image $32 \times 32$ is flattened into $N = 1024$ tokens.
- Each token initially contains RGB values.

### 2) Fourier positional encoding (2D)
Flattening removes explicit 2D structure, so we inject location information using Fourier features.

For a pixel at normalized coordinate $(x, y)\in[-1,1]^2$ and frequency $f_k$, the feature block is:

$$sin(\pi f_k x), \cos(\pi f_k x), \sin(\pi f_k y), \cos(\pi f_k y)$$

We concatenate multiple bands $$k = 1..B$$ and also include raw coordinates $[x, y]$.

With `num_bands = 16`, the positional dimension is:

$$\mathrm{pos\_dim} = 2 + 4B = 2 + 64 = 66$$

So each pixel token becomes $[RGB(3) \ \| \ pos(66)] \Rightarrow 69$ dimensions before projection.

### 3) Cross-attention bottleneck (input → latents)
The core Perceiver mechanism uses asymmetric attention:
- **Queries** from a learned latent array (size $M$)
- **Keys/Values** from input tokens (size $N$)

This compresses a potentially huge input into a fixed-size latent representation.

### 4) Latent Transformer tower (latents ↔ latents)
After cross-attention, we refine representations using standard self-attention blocks **only over latents** (size $M$), independent of $N$.

### 5) Classification head
For CIFAR-10 classification, we use:
- Mean pooling over latents $\rightarrow$ a single vector
- LayerNorm + Linear projection to 10 logits

---

## Architecture Overview
Given an image $x \in \mathbb{R}^{B \times 3 \times H \times W}$:

1. Flatten pixels into $N=H\cdot W$ tokens
2. Compute Fourier 2D positional encoding (shape $N \times \mathrm{pos}_{\mathrm{dim}}$)
3. Concatenate $[RGB \| pos]$ and project to $d_{\text{model}}$
4. Expand learned latents to batch size
5. Cross-attention: latents read input tokens
6. Latent self-attention tower: refine latents
7. Pool latents and classify

---

## Mathematical Formulation

### Attention
Scaled dot-product attention:

$$\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

In **cross-attention**:
- $Q$ from latents $Z \in \mathbb{R}^{M \times d}$
- $K,V$ from input tokens $X \in \mathbb{R}^{N \times d}$

In **self-attention**:
- $Q,K,V$ from latents only

### Fourier features
For 2D coordinates $(x,y)$ and a set of frequencies $\{f_k\}_{k=1}^{B}$:

$$\gamma(x,y) = [x,y,\ \sin(\pi f_1 x),\cos(\pi f_1 x),...,\sin(\pi f_B x),\cos(\pi f_B x),\ \sin(\pi f_1 y),\cos(\pi f_1 y),...,\sin(\pi f_B y),\cos(\pi f_B y)]$$

### Complexity
- Vanilla Transformer attention on input tokens:

$$O(N^2)$$

- Perceiver cross-attention:

$$O(NM)$$

- Latent self-attention:

$$O(M^2)$$

With $M \ll N$, this is significantly more scalable.

---

## Implementation Notes
**Model settings used in this notebook:**
- `num_latents = 128`
- `d_model = 256`
- `num_heads = 8`
- `num_self_attn_blocks = 4`
- `dropout = 0.1`
- `num_bands = 16`, `max_freq = 16.0`
- Input tokens: $N = 32 \cdot 32 = 1024$
- Positional dim: 66, token dim before projection: 69

Parameter count printed by the notebook:
- ~**4.00M** parameters

---

## Training Setup
- Dataset: **CIFAR-10**
- Train augmentation:
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip()
- Optimizer: **AdamW**
  - learning rate: `3e-4`
  - weight decay: `1e-2`
- Batch size: `128`
- Epochs: `10`
- Loss: cross-entropy

---

## Results
Final epoch metrics from the run in this repository:

- **Train loss:** 1.3437  
- **Train accuracy:** 51.36%  
- **Validation loss:** 1.2618  
- **Validation accuracy:** 54.19%

The training curves show a steady decrease in loss and increase in accuracy across epochs, indicating stable optimization and correct integration of the Perceiver components.

---

## Discussion
This reproduction demonstrates that a **convolution-free**, token-based Perceiver architecture can be trained end-to-end on CIFAR-10 using Fourier positional features and a latent bottleneck.

Key observations:
- **Stable learning:** both train and validation loss decrease consistently across epochs.
- **Validation slightly above training:** plausible due to training-time augmentation and dropout (evaluation runs without augmentation and dropout is disabled).
- **Moderate accuracy:** expected for a minimal Perceiver on CIFAR-10. CNNs typically perform better quickly due to strong image inductive biases (locality, translation equivariance). This model must learn such structure via attention + positional encodings.

The primary goal here is architectural clarity: showing how cross-attention compresses high-dimensional inputs into a fixed latent space, after which computation is independent of input length.

---

## Reproducibility
The notebook sets random seeds for Python, NumPy, and PyTorch. Exact determinism on GPU is not guaranteed due to non-deterministic kernels, but results should be broadly repeatable.

Environment prints include:
- `torch.__version__`
- CUDA device name

---

## Limitations & Future Work
This is a minimal reproduction designed for clarity rather than state-of-the-art performance. Potential extensions:
- Add CIFAR-10 channel normalization (mean/std)
- Train longer and/or add a learning rate schedule
- Use log-spaced frequency bands (e.g., $2^k$)
- Use a stronger decoder (attention pooling or Perceiver IO-style output queries)
- Use separate LayerNorms for latent queries and input K/V in cross-attention
- Evaluate scaling on higher-resolution images or larger datasets to highlight compute advantages

---

## Citation
If you use this project, please cite the original paper:

**Jaegle et al., 2021** — *Perceiver: General Perception with Iterative Attention*  
https://arxiv.org/abs/2103.03206

---