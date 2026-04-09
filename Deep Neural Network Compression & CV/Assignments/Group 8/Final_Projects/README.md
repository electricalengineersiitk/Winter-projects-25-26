# Deep Compression MLP

A PyTorch implementation of the **Deep Compression** pipeline applied to a Multi-Layer Perceptron (MLP) trained on the MNIST dataset. This project demonstrates the full compression workflow — pruning, quantization, and Huffman encoding — achieving a **31.3× reduction** in model size with minimal accuracy loss.

---

## Overview

Deep Compression is a model compression technique introduced by Song Han et al. (ICLR 2016 Best Paper). It combines three sequential stages to dramatically reduce neural network storage requirements:

1. **Pruning** — Remove redundant weights via magnitude-based thresholding
2. **Quantization** — Share weights using K-Means clustering (8-bit)
3. **Huffman Encoding** — Entropy-encode the quantized weight indices

This repo applies that pipeline end-to-end to an MLP classifier on MNIST.

---

## Results

| Stage | Accuracy | Model Size | Compression |
|---|---|---|---|
| Baseline (32-bit) | 97.33% | 1.7793 MB | 1× |
| After Pruning + Fine-tune | 97.71% | — | ~2× (51.67% sparse) |
| After Quantization (8-bit) | 97.70% | 0.4448 MB | **4×** |
| After Huffman Encoding | 97.35% | 0.0568 MB | **31.3×** |

> Final model is **31.3× smaller** than the original with only **~0%** accuracy drop.

---

## Pipeline Breakdown

### Phase 1 — Baseline Training

An MLP (`MNIST_MLP`) is trained on MNIST for 5 epochs using standard cross-entropy loss.

```
Epoch 1  →  Loss: 0.4633  |  Acc: 94.41%
Epoch 2  →  Loss: 0.1458  |  Acc: 95.16%
Epoch 3  →  Loss: 0.0974  |  Acc: 96.75%
Epoch 4  →  Loss: 0.0713  |  Acc: 96.58%
Epoch 5  →  Loss: 0.0569  |  Acc: 97.33%
```

---

### Phase 2 — Pruning

**Magnitude Pruning** zeros out the smallest-magnitude weights globally across all layers.

- Pre-pruning sparsity: 50.00% (structured masks applied)
- Post-pruning sparsity: **51.67%** (240,467 weights zeroed out of 465,408)

After pruning, the model is **fine-tuned for 3 epochs** to recover accuracy:

```
[Finetune] Epoch 1  →  Loss: 0.0321  |  Acc: 97.63%
[Finetune] Epoch 2  →  Loss: 0.0240  |  Acc: 97.72%
[Finetune] Epoch 3  →  Loss: 0.0189  |  Acc: 97.71%
```

---

### Phase 3 — Quantization (K-Means Weight Sharing)

Active weights in each layer are clustered into **256 unique values** (8-bit) using K-Means, replacing full-precision floats with shared codebook indices.

| Layer | Unique Active Weights |
|---|---|
| `classifier.0` | 232 |
| `classifier.2` | 211 |
| `classifier.4` | 195 |
| `classifier.6` | 195 |

**Storage savings:**

| | Size |
|---|---|
| Baseline (32-bit float) | 1.7793 MB |
| Quantized (8-bit) | 0.4448 MB |
| **Compression ratio** | **4.0×** |

Post-quantization accuracy: **97.35%**

---

### Phase 4 — Huffman Encoding

Huffman encoding is applied to the quantized weight indices, assigning shorter bit codes to more frequent values and longer codes to rare ones.

```
Unique Weight Clusters : 1,035
Most frequent value    : 0.0000  →  Freq: 465,408  →  Code: "1" (1 bit)
Rare weight example    : -0.0165 →  Freq: 1         →  Code: 11 bits
```

**Final encoded size: 0.0568 MB → 31.3× smaller than baseline**

---

## Model Architecture

```
mnist (custom wrapper)
└── Flatten
└── Sequential (classifier)
    ├── modified_linear   ← compression-aware linear layer
    ├── ReLU
    ├── modified_linear
    ├── ReLU
    ├── modified_linear
    ├── ReLU
    └── modified_linear

Total Parameters: 466,442
```

The `modified_linear` module (in `compression/linear.py`) extends `nn.Linear` to support pruning masks and quantized weight lookup.

---

## Project Structure

```
deep-compression-mlp/
├── main.py                  # Entry point — runs the full compression pipeline
├── models/
│   └── mnist.py             # MNIST MLP model definition
└── compression/
    └── linear.py            # modified_linear: pruning + quantization-aware layer
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn (for K-Means quantization)
- numpy

Install dependencies:

```bash
pip install torch torchvision scikit-learn numpy
```

### Run

```bash
python main.py
```

The script will automatically:
1. Download the MNIST dataset
2. Train the baseline MLP
3. Apply pruning + fine-tuning
4. Apply 8-bit K-Means quantization
5. Apply Huffman encoding
6. Print a full compression report

---

## Device Support

The pipeline detects and runs on the available device:

```
[*] Computation Device: cpu
```

GPU (CUDA) is used automatically if available.

---

## References

- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) — Song Han, Huizi Mao, William J. Dally (ICLR 2016, Best Paper Award)

---

## License

This project is for educational and research purposes.
