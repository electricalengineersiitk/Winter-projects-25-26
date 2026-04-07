# Deep Neural Network Compression for Computer Vision

## Overview

This project implements a deep learning pipeline for image
classification and model compression. A Convolutional Neural Network
(CNN) is trained on a large-scale fruit classification dataset and then
compressed using a three-stage pipeline.

## Objective

-   Train a CNN for multi-class fruit classification

-   Apply deep compression techniques

-   Reduce model size while maintaining accuracy

## Project Structure

    /models             -> Trained models
    /compression         -> Compression algorithms
    /compressed_models   -> Final compressed models
    /utils               -> Helper functions
    /data/dataset        -> Training and test data

## Requirements

-   Python 3.8+

-   PyTorch

-   NumPy

-   scikit-learn

Install dependencies:

    pip install torch torchvision numpy scikit-learn

## How to Run

### Step 1: Setup

Ensure dataset is placed inside:

    data/dataset/Training
    data/dataset/Test

### Step 2: Train Model

Run the training script or notebook:

    python main.py

### Step 3: Compression Pipeline

Run compression modules sequentially:

-   Pruning: removes redundant weights (90% sparsity)

-   Quantization: clusters active weights into K=16 distinct centroids

-   Huffman coding: losslessly compresses the final representations

## Model Details

-   Architecture: CNN (VGG-style)

-   Dataset: Fruits-360

-   Classes: 257

-   Batch Size: 64

## Output

The system produces:

-   Trained baseline model (.pth file)

-   Compressed model (.npz file)

-   Loss values and accuracy metrics during training

## Results Summary

The 3-stage compression pipeline successfully achieved an extreme
compression ratio while the pruning phase simultaneously acted as a
regularizer, preventing overfitting and actively improving the model's
predictive accuracy.

-   **Disk Space:** 37.57 MB (Baseline) → 0.61 MB
    (Compressed)

-   **Compression Ratio:** 61.6 times

-   **Test Accuracy:** 90.11% (Baseline) → 93.26%
    (Compressed)

-   **Accuracy Gain:** +3.15%

-   **Average Bits/Weight:** ~0.52 bits (down from
    32.0 FP32)

-   **Peak Runtime Memory:** 208.54 MB → 241.29 MB (Expected
    overhead for on-the-fly reconstruction)

## Notes

-   GPU is highly recommended for faster training and clustering.

-   This extreme compression makes the final model highly efficient for
    storage and deployment on resource-constrained edge devices.
