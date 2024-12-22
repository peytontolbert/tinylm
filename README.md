# Language Model Training System

A PyTorch-based implementation of a language model training system with Mixture of Experts (MoE) capability. This system is designed to train models on text data with efficient distributed computation and memory usage.

## Features

- Bigram Language Model implementation
- Mixture of Experts (MoE) architecture support
- CUDA-accelerated training with mixed precision
- Customizable hyperparameters
- Training and validation loss tracking
- Checkpoint saving and loading
- Text generation capabilities

## Requirements

- Python 3.7+
- PyTorch
- CUDA-capable GPU (recommended)

## Project Structure

```
├── train.py # Main training script
├── eval.py # Model evaluation script
├── moe.py # Mixture of Experts implementation
├── utils.py # Utility functions
├── BigramLanguageModel.py # Core language model implementation
└── README.md # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/peytontolbert/tinylm.git
```

2. Install the required dependencies:

```bash
pip install torch
```

## Usage

### Training

To train the model:
```bash
python train.py
```

Key hyperparameters in `train.py`:
- `batch_size`: Number of sequences processed in parallel
- `block_size`: Maximum context length for predictions
- `learning_rate`: Initial learning rate
- `max_iters`: Total training iterations

### Evaluation

To evaluate the model:
```bash
python eval.py
```

### Model Architecture

The system includes:
- Bigram Language Model for token prediction
- Optional Mixture of Experts layer for improved model capacity
- Gradient scaling for mixed precision training
- Learning rate scheduling with warmup

## Checkpoints

The system automatically saves checkpoints during training:
- Every 1000 steps
- At the end of training
- Includes model state and vocabulary mappings