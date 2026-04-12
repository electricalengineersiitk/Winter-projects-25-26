# Deep Neural Network Compression & CV

Implementation of the **Deep Compression** pipeline from:

> **"Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding"**  
> Song Han, Huizi Mao, William J. Dally  
> ICLR 2016 ([arXiv:1510.00149](https://arxiv.org/abs/1510.00149))

## Project Goal

Build a 3-stage compression pipeline for a VGG-style CNN trained on CIFAR-10:
- **Target compression**: ~9x (from ~37 MB to ~4 MB)
- **Max accuracy drop**: < 1.5%
- **Average bits per weight**: ~3.57

## Pipeline Overview

```
Original Model → [Stage 1: Pruning] → [Stage 2: Quantization] → [Stage 3: Huffman] → Compressed Model
   ~37 MB            90% sparse            K-Means clusters          Lossless           ~4 MB
                     (9-13x)               (27-31x total)          (35-49x total)
```

### Stage 1: Network Pruning (Section 2)
- Remove weights below a magnitude threshold (target: 90% sparsity)
- Retrain ("fine-tune") the surviving weights to recover accuracy
- Store sparse structure using CSR/CSC format with relative indices

### Stage 2: Trained Quantization / Weight Sharing (Section 3)
- K-Means clustering of non-zero weights per layer
  - CONV layers: 256 clusters (8-bit indices)
  - FC layers: 32 clusters (5-bit indices)
- Linear initialization of centroids (paper's best method)
- Fine-tune centroids via gradient aggregation (Eq. 3 in paper)

### Stage 3: Huffman Coding (Section 4)
- Variable-length encoding of quantized weight indices
- Exploits biased distribution of cluster labels
- Saves additional 20-30% on top of pruning + quantization
- Applied offline — no training needed

## Project Structure

```
Project/
├── config.py                  # All hyperparameters & paths
├── main.py                    # End-to-end pipeline execution
├── data/
│   └── data_loader.py         # CIFAR-10 loading + augmentation
├── models/
│   ├── __init__.py
│   └── vgg_cifar.py           # VGG-11 adapted for CIFAR-10
├── compression/
│   ├── __init__.py
│   ├── pruning.py             # Stage 1: Magnitude-based pruning
│   ├── quantization.py        # Stage 2: K-Means weight sharing
│   └── huffman.py             # Stage 3: Huffman coding
├── utils/
│   ├── __init__.py
│   └── metrics.py             # Accuracy, sparsity, size metrics
└── compressed_models/         # Saved model checkpoints & results
```

## Requirements

```
torch >= 1.10
torchvision
numpy
scikit-learn
```

Install:
```bash
pip install torch torchvision numpy scikit-learn
```

## Usage

### Full pipeline (train + compress):
```bash
cd Project
python main.py
```

### Skip training (load saved baseline):
```bash
python main.py --skip-training
```

### Custom settings:
```bash
python main.py --epochs 50 --prune-sparsity 0.85
```

## Key Paper References

| Section | Topic | Implementation |
|---------|-------|---------------|
| §2 | Network Pruning | `compression/pruning.py` |
| §3.1 | Weight Sharing via K-Means | `compression/quantization.py` |
| §3.2 | Linear Centroid Initialization | `compression/quantization.py` |
| §3.3 | Centroid Fine-tuning (Eq. 3) | `compression/quantization.py` |
| §4 | Huffman Coding | `compression/huffman.py` |
| §5 | Experiments / Evaluation | `utils/metrics.py` + `main.py` |



