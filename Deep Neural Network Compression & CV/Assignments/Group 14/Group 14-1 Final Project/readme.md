Note: Since it will take a long time to train the model while pruning,the compressed model .npz file has been saved in compressed models folder, and the output of running main.py has been attached in output.txt

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
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy scikit-learn matplotlib
```

## Running the Pipeline

### Full Pipeline Execution
`python main.py`

Runs the complete workflow:
- Model training  
- Iterative pruning  
- Fine-tuning  
- K-Means quantization  
- Compression and saving  
- Final evaluation  

---

### Run Inference on Compressed Model
`python deploy.py`

This will:
- Load the compressed model  
- Reconstruct weights  
- Perform a sample forward pass  

---

### Optional: K-Means Cluster Analysis
`python analysis/kmeans_elbow.py`

Used to determine an appropriate number of clusters for quantization.

---

## Project Structure

    .
    ├── models/
    │   └── model.py              # VGG-style CNN architecture
    │
    ├── data/
    │   └── data_loader.py        # CIFAR-10 loading and preprocessing
    │
    ├── utils/
    │   ├── train.py              # Training and evaluation loops
    │   └── metrics.py            # Sparsity and compression statistics
    │
    ├── compression/
    │   ├── prune.py              # Iterative L1 pruning logic
    │   ├── rewind.py             # Weight rewinding after pruning
    │   ├── quantize.py           # K-Means quantization functions
    │   └── huffman.py            # Compression and storage utilities
    │
    ├── deployment/
    │   └── load_model.py         # Load compressed model and run inference
    │
    ├── analysis/
    │   └── kmeans_elbow.py       # Determine optimal number of clusters
    │
    ├── compressed_models/        # Stores compressed .npz model files
    ├── config.py                 # Hyperparameters and device config
    ├── main.py                   # End-to-end pipeline execution
    ├── deploy.py                 # Deployment and inference script

---

## Loading Compressed Model

To load and test the compressed model:

`python deploy.py`

Or programmatically:

    from deployment.load_model import load_compressed_model, run_inference

    model = load_compressed_model("compressed_models/vgg_compressed.npz")
    run_inference(model)

---

## Results

| Metric | Value |
|--------|------|
| Original Model Size | 16.31 MB |
| Compressed Size | 1.59 MB |
| Compression Ratio | 10.28× |
| Accuracy Drop | < 2% |
| Bits per Weight | ~3 |

---

## Notes

- Training can take significant time on CPU (it took me 2+ hours :cry:
- For quick testing, reduce epochs in `config.py`  
- Ensure the output directory exists:  
  `compressed_models/vgg_compressed.npz`  

---

## Result

- Pruning implemented  
- Quantization integrated  
- Compression functional  
- Deployment verified  



