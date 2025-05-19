# Multi-Image SigLIP (multi-im-siglip)

A PyTorch implementation of the SigLIP (Sigmoid Loss for Language-Image Pre-training) loss function with ambiguity resolution for multiple images associated with the same text.

## Overview

This repository provides an implementation of the SigLIP loss function, extended to handle the real-world scenario where multiple images can be associated with the same text description. The implementation includes an ambiguity resolution mechanism that selects the most appropriate image for each text sample during training.

## Features

- **SigLIP Loss Implementation**: Core implementation of the Sigmoid Loss for Language-Image Pre-training
- **Ambiguity Resolution**: Mechanism to select the best image for each text when multiple images are associated with the same text
- **Trainable Parameters**: Temperature and bias parameters are trainable through backward propagation
- **TorchScript Support**: JIT-compiled implementation for faster inference and training
- **Sanity Check Utilities**: Comprehensive logging and verification steps to ensure the implementation behaves as expected

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.7.0
- NumPy ≥ 2.2.6

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/multi-im-siglip.git
cd multi-im-siglip
uv sync
```

## Usage

### Training Example

The repository includes a demonstration training script that shows how to use the loss function in a training loop:

```python
from multi-im-siglip.train import main_training_loop

# Run the training loop with default parameters
main_training_loop()
```

### Basic Usage

For more direct usage of the loss function:

```python
import torch
from multi-im-siglip.main import SigLipLossWithAmbiguityResolution

# Initialize the loss function
loss_fn = SigLipLossWithAmbiguityResolution()

# Example inputs
img_emb = torch.randn(16, 256)  # 16 images with 256-dim embeddings
txt_emb = torch.randn(10, 256)  # 10 text samples with 256-dim embeddings
key = torch.randint(0, 10, (16,))  # Each image is associated with one of the text samples
t_prime = torch.tensor(0.0)  # Temperature parameter (log scale)
b = torch.tensor(0.0)  # Bias parameter

# Compute the loss
final_loss, zimg_selected, ztxt, final_logits, selected_indices = loss_fn.compute_loss(
    img_emb, txt_emb, key, t_prime, b
)
```

## How It Works

The SigLIP loss with ambiguity resolution works in three main steps:

1. **Preprocessing**: L2 normalize image and text embeddings
2. **Ambiguity Resolution**: For each text sample, select the image that has the lowest potential loss among all images associated with that text
3. **Loss Computation**: Apply the standard SigLIP loss function to the selected image-text pairs

### Key Implementation Details

- The loss function uses a trainable temperature parameter `t` (stored as `log_t` for numerical stability) and a bias parameter `b`
- For each text sample, we find all images associated with it using the `key` tensor
- We select the image with the lowest potential loss (best alignment) for each text
- The final loss is computed using the standard SigLIP formulation but with only the selected image-text pairs

## Files

- `main.py`: Contains the core implementation of the SigLIP loss with ambiguity resolution
- `train.py`: Provides a full training loop example with dummy models and data

## License

[Your license here]

## Citation

If you use this code in your research, please cite:

```
@misc{multi-im-siglip,
  author = {Your Name},
  title = {Multi-Image SigLIP: SigLIP Loss with Ambiguity Resolution},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/multi-im-siglip}}
}
```

## Acknowledgements

This implementation is inspired by the SigLIP paper:
- [SigLIP: Sign-to-Language Image Pre-training for Vision-Language Models](https://arxiv.org/abs/2303.15343)
