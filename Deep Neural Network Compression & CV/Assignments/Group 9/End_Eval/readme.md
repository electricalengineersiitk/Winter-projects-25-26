### Deep Neural Network Compression Pipeline

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

##  Project Overview
This repository contains a high-performance **Deep Model Compression Pipeline** built to reduce the storage footprint and memory overhead of neural networks. By integrating custom mathematical layers, we have enabled a compression strategy that shrinks deep learning models by over **6x** while maintaining—and even improving—baseline accuracy.

### Our Core Architectures
We demonstrated this compression framework on two distinct architectures:
1.  **FruitsFusionNet:** A custom multi-modal architecture that fuses standard images with LBP texture, Canny edges, shape, and color data to classify 131 fruit varieties.
2.  **SmallCIFARNet:** A 5-layer Convolutional Neural Network designed for efficient image classification on the CIFAR-10 dataset.

---

##  Methodology: How We Built This

Our pipeline modifies the standard PyTorch forward-pass workflow through four key technological pillars:

### 1. Magnitude Pruning (`prune.py`)
We implemented custom `modified_conv2d` and `modified_linear` layers that support a distinct **Prune Mode**. The algorithm identifies the "weakest" parameters based on their absolute magnitude and dynamically calculates a threshold $\tau$. Weights below this threshold are masked to zero, eliminating redundant connections and simplifying the network's internal logic.

### 2. K-Means Quantization (`quantization.py`)
Instead of storing every remaining weight as a heavy 32-bit floating-point number, we apply **K-Means Clustering** to group the non-zero weights into $k=16$ distinct clusters. Each weight is then replaced by a 4-bit index pointing to a shared "codebook" of 16 centroids, drastically reducing the precision overhead.

### 3. Sparse CSR Storage (`loading.py`)
Standard PyTorch `.pth` files save all values, including the zeros generated during pruning, negating any storage benefits. We built a custom saver that intercepts the quantized weights and encodes them using a **Compressed Sparse Row (CSR)** format via `scipy.sparse`. This allows us to save only the meaningful, non-zero indices into a highly compressed `.npz` file.

### 4. Entropy-Based Compression (Huffman Encoding)
To squeeze the absolute maximum compression out of the `.npz` files, our pipeline utilizes the DEFLATE algorithm inherent in NumPy's `np.savez_compressed()` function. Because K-Means quantization naturally creates non-uniform distributions (certain weight clusters appear much more frequently than others), this underlying algorithm inherently applies **Huffman Encoding**. It assigns shorter bit-lengths to the most frequent weight indices, resulting in significant, lossless entropy compression on top of our sparse CSR format.

---

## 📂 Repository Structure

```text
├── compression/
│   ├── __init__.py
│   ├── conv2d.py           # Custom Conv2d layer with prune/quantize modes
│   ├── linear.py           # Custom Linear layer with prune/quantize modes
│   ├── prune.py            # Magnitude pruning logic and sparsity verification
│   └── quantization.py     # K-Means clustering logic for weight quantization
├── models/
│   ├── model_fruits.py     # Multi-modal FruitsFusionNet architecture
│   └── model_cifar.py      # SmallCIFARNet architecture
├── data/
│   └── data_loader.py      # Automated dataset downloading (Kagglehub) and feature extraction
├── config.py               # Hardware device configuration (CPU/CUDA)
├── loading.py              # CSR-based sparse saving and `.npz` loading utilities
├── main.py                 # Core execution and pipeline orchestration script
└── README.md               # Project documentation
```

---

## User Manual: Running Instructions
## 1. Setup & Installation
Ensure you have Python 3.8+ installed. Install the required dependencies to run the pipeline:

```Bash
pip install torch torchvision numpy scipy scikit-learn opencv-python kagglehub

```

## 2. Execute the Full Pipeline
To automatically download the dataset, train the baseline model, apply magnitude pruning, apply K-Means quantization, and export the Huffman-encoded sparse .npz file, run the orchestration script:

```Bash
python main.py

```

## 3. Loading the Compressed Model for Inference
Because the model is saved using custom CSR sparse encoding, you cannot use standard PyTorch loaders. To deploy the compressed weights for inference:

```Python
from models.model_fruits import FruitsFusionNet
from loading import load_model_from_npz
from config import config_device

device = config_device()

# 1. Initialize an empty version of the architecture
model = FruitsFusionNet().to(device)

# 2. Reconstruct the full weights from the sparse codebook
model = load_model_from_npz(model, 'compressed_models/compressed.npz', device)

# 3. Model is now fully restored and ready for standard inference
# output = model(image, lbp, canny, shape, color)

```
---

## Experimental Results
Below is the step-by-step breakdown of our experimental results on FruitsFusionNet.

## Phase 1: Baseline Training
The starting point. We trained the standard, uncompressed model from scratch so we had a baseline to compare against.

- Process: The model was trained over 15 epochs. As it learned, the "Loss" (error rate) steadily dropped from 3.5651 down to a highly stable 1.5166.

- Baseline Accuracy: 92.05%

## Phase 2: Magnitude Pruning
The first compression step. Think of this like trimming dead branches off a tree. We forced the model to delete the weakest, least important connections (weights).

- Sparsity Achieved: 50.00% (Exactly half of the model's weights were safely removed).

- Fine-Tuning: We trained the pruned model for 5 more epochs to let it heal and adjust to the missing weights.

- Pruned Accuracy: 93.75% * Insight: Removing 50% of the weights actually improved the accuracy! By eliminating the noisy, weak connections, the model became more focused and generalized better.

## Phase 3: K-Means Quantization
The second compression step. Instead of letting every remaining weight be a unique, heavy 32-bit decimal number, we forced them to share a small "palette" of numbers.

- Clusters (k): 16 (The entire model now only uses 16 unique numbers to represent all of its weights).

- Fine-Tuning: 5 epochs of training to adjust to this new restricted palette.

- Quantized Accuracy: 92.86%

- Insight: Despite stripping away 50% of the architecture and massively reducing the mathematical precision of the remaining weights, the model successfully maintained its predictive power.

---

#  Final Compression Report & Verification

After pruning and quantizing, we saved the model using a custom **Compressed Sparse Row (CSR)** format. By passing this through `np.savez_compressed()`, the data was further subjected to **Huffman Encoding (via DEFLATE)**, successfully exploiting the non-uniform weight distributions created during **K-Means clustering**.

This resulted in a **tiny `.npz` file**, completely avoiding the storage of millions of useless zeroes.

---

##  Verification

We loaded the exact compressed file back into a blank model to ensure the integrity of the pipeline.

**Final Decompressed Model Accuracy:**  
`92.86%` 

---

##  Final Comparison

| Metric                  | Original Standard Model | Our Custom Pipeline |
|------------------------|------------------------|---------------------|
| **Disk Size**          | 8.45 MB                | 1.32 MB             |
| **Accuracy**           | 92.05%                 | 92.86%              |

---

##  Hardware Memory Usage

- **CUDA Memory Allocated:** 44.69 MB  
- **CUDA Memory Reserved:** 344.00 MB  

---

##  Conclusion

Our compression framework was a **complete success**.

By combining:
- Magnitude Pruning  
- K-Means Quantization  
- Sparse CSR Representation  
- Huffman-based Entropy Encoding  

we achieved a **6.42× reduction in overall file size**, reducing the model from **8.45 MB → 1.32 MB**.

 **Key Highlight:**  
Despite this massive compression, the model showed a **net increase in accuracy (+0.81%)** compared to the original baseline.

---

##  Summary

-  Significant storage reduction  
-  No data loss after compression/decompression  
-  Improved model accuracy  
-  Efficient deployment-ready format  

---
