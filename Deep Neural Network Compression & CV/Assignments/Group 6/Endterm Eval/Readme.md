#  Deep Compression Pipeline

> A full neural network compression pipeline applied to MNIST digit classification, implementing **Pruning → Quantization → Huffman Encoding** — the three-stage approach inspired by Han et al. (2015).

---

##  Overview

This project demonstrates how a trained neural network (MLP) can be dramatically compressed without meaningful loss in accuracy. The pipeline takes a baseline model, prunes redundant weights, quantizes the remaining ones using weight sharing, and finally applies Huffman encoding to achieve maximum storage reduction.

The entire pipeline is orchestrated via `main.py` and evaluated at each stage.

---

## Project Structure

```
deep-compression/
│
├── compression/
│   ├── pruning.py          # Magnitude-based weight pruning
│   ├── quantization.py     # Weight sharing (k-means quantization)
│   └── huffman.py          # Huffman encoding for final compression
│
├── data/
│   └── dataset.py          # MNIST DataLoader builder
│
├── models/
│   └── mlp.py              # MNIST Multi-Layer Perceptron architecture
│
├── utils/
│   └── training.py         # Training loop & evaluation utilities
│
├── config.py               # Hyperparameters & configuration
├── main.py                 # End-to-end pipeline orchestrator
└── final_training.ipynb    # Jupyter notebook for experiments
```

---

##  Configuration

All key hyperparameters are centralized in `config.py`:

| Parameter            | Value   | Description                                      |
|---------------------|---------|--------------------------------------------------|
| `BATCH_SIZE`         | 128     | Number of samples per training batch             |
| `LEARNING_RATE`      | 0.001   | Adam optimizer learning rate                     |
| `EPOCHS`             | 5       | Training epochs for baseline model               |
| `PRUNING_THRESHOLD`  | 0.5     | Magnitude threshold — weights below this are zeroed |
| `QUANTIZATION_BITS`  | 8       | Bit-width for weight sharing clusters            |
| `DEVICE`             | auto    | Uses CUDA if available, otherwise CPU            |

---

##  Pipeline: 4 Phases

### Phase 1 — Baseline Training

The MNIST dataset is loaded and an MLP model is trained from scratch for 5 epochs.

- **Dataset:** MNIST (60,000 training / 10,000 test images)
- **Model:** Fully connected MLP (`784 → hidden → 10`)
- **Optimizer:** Adam (`lr = 0.001`)
- **Goal:** Establish a strong accuracy benchmark before any compression

---

### Phase 2 — Magnitude Pruning

Weights with absolute value below the pruning threshold (`0.5`) are zeroed out. The model is then fine-tuned so it can "heal" around the remaining connections.

**Steps:**
1. Print sparsity stats **before** pruning
2. Apply magnitude pruning masks
3. Print sparsity stats **after** pruning (to see exact change)
4. Evaluate dropped accuracy (expected drop before fine-tuning)
5. Fine-tune for `EPOCHS // 2` epochs at `LR / 10`
6. Make pruning permanent (removes PyTorch tracking masks)

**Key metric:** Global sparsity — what percentage of weights are exactly zero.

---

### Phase 3 — Quantization (Weight Sharing)

Instead of storing each weight as a 32-bit float, weight sharing groups all weights into `2^bits` clusters (e.g., 256 clusters for 8-bit). Each weight is replaced by the index to its nearest cluster centroid.

- **Technique:** k-means weight sharing
- **Bits:** 8-bit (256 clusters per layer) — configurable via `QUANTIZATION_BITS`
- **Effect:** Massively reduces the number of unique values stored per layer

**Unique weights before quantization:**
| Layer | Unique Active Weights |
|-------|-----------------------|
| fc1   | 100,199               |
| fc2   | 16,381                |
| fc3   | 640                   |

**Unique weights after 8-bit quantization:**
| Layer | Unique Active Weights |
|-------|-----------------------|
| fc1   | 256                   |
| fc2   | 256                   |
| fc3   | 256                   |

---

### Phase 4 — Huffman Encoding

Huffman encoding is applied as the final compression step. More frequent weight indices get shorter binary codes, further reducing the bits needed to store the model.

- **Input:** Quantized weight indices from Phase 3
- **Output:** Variable-length binary encoding
- **Result:** `huffman_size_mb` — the final on-disk model size

---

##  Results Summary

| Stage                       | Test Accuracy | Model Size        | Notes                                      |
|-----------------------------|---------------|-------------------|--------------------------------------------|
| Baseline (5 epochs)         | **96.91%**    | 0.8955 MB (32-bit)| Full float32 weights, 234,752 total        |
| After Pruning (immediate)   | 96.70%        | —                 | 117,376 weights zeroed (50% sparsity)      |
| After Fine-Tuning (2 epochs)| **97.88%**    | sparse            | Accuracy recovered — even beats baseline!  |
| After 8-bit Quantization    | **97.89%**    | 0.1119 MB (8-bit) | 256 unique weights per layer (8.0x smaller)|
| After Huffman Encoding      | **97.89%**    | 0.1100 MB         | Variable-bit codes, **8.1x final ratio**   |

### Training Progress (Baseline)

| Epoch | Loss   | Train Accuracy |
|-------|--------|----------------|
| 1/5   | 0.3991 | 88.05%         |
| 2/5   | 0.1764 | 94.68%         |
| 3/5   | 0.1267 | 96.13%         |
| 4/5   | 0.0991 | 96.91%         |
| 5/5   | 0.0804 | 97.56%         |

### Storage Savings

| Model Version     | Size (MB) | Bits  |
|-------------------|-----------|-------|
| Baseline          | 0.8955    | 32-bit float |
| After Quantization| 0.1119    | 8-bit |
| After Huffman     | 0.1100    | Variable-bit |

> **Final Compression Ratio: 8.1× smaller** with only a **+0.98% accuracy gain** over baseline!

### Huffman Encoding Snapshot

768 unique weight clusters were encoded. Sample codes:

| Weight Value | Frequency | Huffman Code  |
|-------------|-----------|---------------|
| 0.0315      | 1,632     | `011111` (6 bits) |
| 0.0307      | 1,615     | `011101` (6 bits) |
| -0.0272     | 1,578     | `011001` (6 bits) |
| -0.0255     | 1,571     | `011000` (6 bits) |
| 0.0298      | 1,560     | `010110` (6 bits) |

**Final Pipeline Output:**
```
[*] Original Baseline Accuracy: 96.91%
[*] Pruned (Fine-Tuned) Acc:    97.88%
[*] Final Quantized Accuracy:   97.89%
[*] Total Network Sparsity:     50.00%

[*] --- STORAGE SAVINGS ---
   [-] Baseline Storage Size:    0.8955 MB (32-bit)
   [-] Quantized Storage Size:   0.1119 MB (8-bit)
   [-] Huffman Encoded Size:     0.1100 MB (Variable-bit)

   [!!!] FINAL COMPRESSION RATIO: 8.1x smaller [!!!]
```

---

##  How to Run

### 1. Install Dependencies

```bash
pip install torch torchvision
```

### 2. Run the Full Pipeline

```bash
python main.py
```

### 3. Run via Notebook

Open `final_training.ipynb` in Jupyter for an interactive step-by-step walkthrough.

---

##  Key Techniques

| Technique              | Module                    | Description                                            |
|------------------------|---------------------------|--------------------------------------------------------|
| Magnitude Pruning      | `compression/pruning.py`  | Zeroes out weights below a threshold                   |
| Weight Sharing         | `compression/quantization.py` | k-means clustering to reduce unique weight values  |
| Huffman Encoding       | `compression/huffman.py`  | Variable-length encoding of quantized indices          |
| Fine-tuning            | `utils/training.py`       | Recover accuracy lost during pruning                   |

---

##  Reference

This project is based on the paper:

> **Han, S., Mao, H., & Dally, W. J. (2015).** *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR 2016. [arXiv:1510.00149](https://arxiv.org/abs/1510.00149)

---

