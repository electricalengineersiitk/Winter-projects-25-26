# CNN Compression Pipeline for CIFAR-10

A complete implementation of a 3-stage model compression pipeline for a VGG-style convolutional neural network trained on CIFAR-10. The objective is to significantly reduce model size while preserving accuracy, enabling efficient deployment on limited-resource systems.

---

## Pipeline Summary

The workflow follows a structured sequence:

1. Train baseline CNN  
2. Apply iterative pruning  
3. Fine-tune sparse model  
4. Perform K-Means quantization  
5. Compress and store model  
6. Reload and validate for inference  

---

## Core Components

### Pruning
- Iterative magnitude-based pruning
- Gradually removes low-importance weights
- Uses binary masks to enforce sparsity
- Incorporates weight rewinding to stabilize training

### Quantization
- Applies K-Means clustering on non-zero weights
- Reduces precision of weights to a smaller set of representative values
- Uses cluster count derived from analysis (e.g., elbow method)
- Current setup uses 8 clusters (3-bit representation)

### Compression
- Stores model using NumPy `.npz` compressed format
- Reduces disk footprint significantly
- Enables fast loading for deployment

### Deployment
- Reconstructs model from compressed weights
- Performs inference to verify correctness
- No retraining required

---

## Installation

### Setup
```bash
git clone <your-repo-link>
cd <project-folder>


