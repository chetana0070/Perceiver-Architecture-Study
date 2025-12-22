# Perceiver Architecture Study

**Independent PyTorch implementation and analysis of the Perceiver architecture**  
(Perceiver: *General Perception with Iterative Attention*, Jaegle et al., DeepMind, ICML 2021)

This repository contains a clean, minimal implementation of the Perceiver architecture, focusing on
its core design principles: **cross-attention bottlenecks**, **latent Transformer processing**, and
**Fourier positional encodings**. The goal is architectural understanding and analysis rather than
full-scale reproduction of the original DeepMind results.

---

## Motivation

Standard Transformers scale quadratically with input size, making them impractical for raw,
high-dimensional inputs such as images, audio waveforms, point clouds, or multimodal data.

The Perceiver addresses this limitation by:
- Introducing a **learned latent bottleneck**
- Using **asymmetric cross-attention** from inputs to latents
- Decoupling **model depth from input dimensionality**

This project studies these ideas through a lightweight, interpretable implementation.

---

## What This Repository Contains

- **Independent Perceiver-style implementation in PyTorch**
- **Fourier positional encodings** for 2D images
- **Cross-attention + latent self-attention stack**
- End-to-end training on **CIFAR-10**
- Training curves, qualitative predictions, and analysis
- A concise written report summarizing insights and limitations

This work is **not a strict reproduction** of the original DeepMind implementation. Instead, it is a
deliberate, minimal implementation designed to validate and study the Perceiver’s architectural
mechanisms.

---

## Repository Structure

```
Perceiver-Architecture-Study/
│
├── data/                          # CIFAR-10 dataset (auto-downloaded)
├── Code Implementation.ipynb      # Main PyTorch implementation notebook
├── Code Implementation (PDF).pdf  # Exported notebook (read-only)
├── Original Paper (PDF).pdf       # Jaegle et al., ICML 2021
├── Reimplementation-Report.pdf    # One-page technical analysis & insights
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── LICENSE
```

---

## Model Overview

**Design choices in this study:**
- Latent array: 128 latents × 256 dimensions
- Single cross-attention block (inputs → latents)
- 4 latent self-attention blocks
- Mean-pooled latents → classification head
- Optimizer: AdamW
- Dataset: CIFAR-10

Despite its simplicity and lack of convolutional priors, the model:
- Trains stably
- Shows smooth loss curves
- Reaches ~52% validation accuracy in 10 epochs

This behavior aligns with claims in the original paper regarding scalability and generality.

---

## Mathematical Models & Formulation

### Problem Setting

Let the input be a high-dimensional array:

$$
X = \{x_i\}_{i=1}^{M}, \quad x_i \in \mathbb{R}^{d_x}
$$

Standard self-attention has quadratic complexity:

$$
\mathcal{O}(M^2)
$$

which becomes infeasible for large inputs.

---

### Latent Bottleneck

The Perceiver introduces a learned latent array:

$$
Z = \{z_j\}_{j=1}^{N}, \quad z_j \in \mathbb{R}^{d_z}, \quad N \ll M
$$

This latent space acts as an information bottleneck.

---

### Cross-Attention (Inputs → Latents)

Asymmetric cross-attention projects input information into the latent space:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

with:
- $Q = ZW_Q$
- $K = XW_K$
- $V = XW_V$

Resulting update:

$$
Z' = \text{CrossAttn}(Z, X)
$$

**Complexity:** $\mathcal{O}(MN)$

---

### Latent Self-Attention

The latents are processed by a Transformer stack:

$$
Z^{(l+1)} = \text{SelfAttn}(Z^{(l)})
$$

**Per-layer complexity:** $\mathcal{O}(N^2)$

---

### Overall Complexity

$$
\mathcal{O}(MN + LN^2)
$$

This formulation decouples model depth $L$ from input size $M$, enabling scalable perception.

---

### Positional Encoding

Since attention is permutation-invariant, spatial structure is injected using Fourier features:

$$
\gamma(x) = [\sin(2^k \pi x), \cos(2^k \pi x)]_{k=1}^{K}
$$

These encodings allow the model to learn spatial relationships without convolutional inductive biases.

---

### Output Mapping

The final representation is obtained via latent aggregation:

$$
y = f\left( \frac{1}{N} \sum_{j=1}^{N} z_j^{(L)} \right)
$$

where $f(\cdot)$ is a task-specific head (classification in this study).

---

## Key Insights

- Latent bottlenecks dramatically reduce attention complexity
- Depth can be increased independently of input size
- Fourier positional encodings provide spatial structure without convolutions
- Early validation accuracy exceeding training accuracy reflects strong regularization
- Accuracy is limited primarily by latent size, depth, and training time

---

## Relevance to Robotics & Control Systems

Understanding scalable perception architectures is critical for:
- Processing raw, high-bandwidth sensory data
- Feeding learned representations into downstream control or RL systems
- Bridging perception and decision-making in autonomous systems

This study complements broader work on controllable perception and perception-to-control pipelines.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Recommended Python version: **Python ≥ 3.9**

---

## References

- Jaegle et al., *Perceiver: General Perception with Iterative Attention*, ICML 2021  
  https://arxiv.org/abs/2103.03206

---

## Author

**Ayushman Mishra**  
Robotics & Control Systems  
https://github.com/aymisxx  
[linkedin.com/in/aymisxx](https://www.linkedin.com/in/aymisxx/)

---