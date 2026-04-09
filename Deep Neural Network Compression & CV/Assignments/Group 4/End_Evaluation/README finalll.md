# Neural Network Compression Pipeline — CIFAR-10 / AlexNet

This project implements a **3-stage deep compression pipeline** applied to a neural network classifier on the CIFAR-10 image dataset. The three stages are **Pruning → Quantization → Huffman Coding**, following the approach from Han et al.'s *Deep Compression* (2015). The goal is to drastically shrink the model's storage size while keeping classification accuracy as high as possible — the kind of compression needed to deploy AI models on phones, drones, or embedded devices.

Here's the simple idea of what happens:
- A frozen, pre-trained **AlexNet** (loaded from PyTorch's model zoo) extracts feature vectors from CIFAR-10 images. You don't train or touch this part.
- A small **MLP (Multi-Layer Perceptron)** sits on top and is the only thing that gets trained and compressed.
- The MLP goes through pruning (removing useless weights), quantization (reducing weight precision), and Huffman coding (lossless bit-level compression).

> **Key constraint:** The CNN (AlexNet) layers are never trained or modified. They are used only as a static feature extractor. All compression is applied only to the MLP classifier.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup — Step by Step](#setup--step-by-step)
3. [Project Structure](#project-structure)
4. [How to Run](#how-to-run)
5. [Expected Terminal Output](#expected-terminal-output)
6. [Loading the Model with Saved Weights](#loading-the-model-with-saved-weights)
7. [Results](#results)
8. [Common Errors and Fixes](#common-errors-and-fixes)
9. [References](#references)

---

## Prerequisites

Before you start, make sure you have the following on your machine:

- **Python 3.8 or higher** — check your version by running `python --version` in your terminal. If you see Python 2.x, use `python3` instead of `python` for all commands below.
- **pip** — Python's package installer, usually comes with Python. Check with `pip --version`.
- **Git** — to clone the repository. Or you can download the ZIP directly from GitHub.
- A working **internet connection** — the CIFAR-10 dataset (~170 MB) and AlexNet pre-trained weights (~230 MB) are downloaded automatically on the first run.

GPU is required otherwise it takes a lot for time to run on 50,000 images. Since none of us had GPU, we ran it on only 500 images(train) and 100 images(test).

---

## Setup — Step by Step

### Step 1: Get the Code

Clone the repository using Git:

```bash
git clone <your-repo-url-here>
cd <project-folder-name>
```

Or if you downloaded a ZIP file, extract it and open a terminal inside the extracted folder.

### Step 2: (Recommended) Create a Virtual Environment

A virtual environment keeps this project's packages isolated from your system Python. This prevents version conflicts with other projects.

```bash
# Create the virtual environment (only do this once)
python -m venv venv

# Activate it — Windows:
venv\Scripts\activate

# Activate it — Mac/Linux:
source venv/bin/activate
```

You'll know it's active when you see `(venv)` at the start of your terminal prompt. Every time you open a new terminal to work on this project, re-run the activate command.

### Step 3: Install Dependencies

```bash
pip install torch torchvision scikit-learn numpy
```

This installs all required libraries. It may take a few minutes depending on your internet speed.

> **Note for GPU users (Nvidia):** The command above installs the CPU-only version of PyTorch by default. For CUDA support, visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and use the install command that matches your CUDA version instead.

### Step 4: Create the Output Folder

The pipeline saves the compressed model into a folder called `compressed_models/`. You need to create it manually before running the code, otherwise you will get a `FileNotFoundError`:

```bash
mkdir compressed_models
```

---

## Project Structure

```
project_root/
│
├── main.py                         # Entry point — run this file to execute the full pipeline
├── config.py                       # Global settings: device selection, batch size, data path, normalization values
│
├── data/
│   ├── __init__.py                 # Makes 'data' a Python package — do not edit or delete
│   └── data_loader.py              # Downloads CIFAR-10 and creates train/test DataLoaders with preprocessing
│
├── models/
│   ├── __init__.py                 # Makes 'models' a Python package — do not edit or delete
│   └── model_cifar.py              # Defines CompressionMLP — the 3-layer MLP that gets trained and compressed
│
├── compression/
│   ├── __init__.py                 # Makes 'compression' a Python package — do not edit or delete
│   ├── linear.py                   # ModifiedLinear — a custom nn.Linear layer that holds pruning masks and quantization state
│   ├── conv2d.py                   # ModifiedConv2d — same idea for Conv2d (present but unused per project constraint)
│   ├── pruning.py                  # prune_model() — zeros out the bottom 90% of weights by magnitude
│   ├── quantization.py             # quantize_model() — clusters weights into K groups using K-Means
│   └── huffman.py                  # Huffman encodes cluster indices for additional lossless compression
│
├── utils/
│   ├── __init__.py                 # Makes 'utils' a Python package — do not edit or delete
│   ├── training.py                 # train_and_eval() — runs AlexNet feature extraction then trains the MLP
│   ├── test_eval.py                # evaluate() — computes test set accuracy
│   └── loading.py                  # save_model_npz() and load_model_from_npz() — compressed weight serialization
│
└── compressed_models/
    └── compressed_mlp.npz          # Created after running main.py — the final compressed model file
```

**What are `__init__.py` files?**
These are empty (or near-empty) files that tell Python "this folder is a package, so you can import from it." You never need to run or edit them — just make sure they exist in their respective folders.

---

## How to Run

### Option 1: Run the Full Pipeline (Recommended for first time)

Make sure your terminal is inside the project root folder (the one containing `main.py`), then run:

```bash
python main.py
```

That single command runs everything end-to-end:

1. Detects your device (GPU/MPS/CPU) and prints which one is being used
2. Downloads CIFAR-10 to `./data_files/` — only on the first run, skipped after that
3. Downloads pre-trained AlexNet weights from PyTorch — only on the first run
4. Trains the baseline MLP for 5 epochs using features extracted by frozen AlexNet
5. Prunes 90% of MLP weights by magnitude and fine-tunes for 2 more epochs
6. Quantizes remaining weights into 16 clusters using K-Means
7. Applies Huffman encoding to further compress the cluster indices
8. Saves the compressed model to `compressed_models/compressed_mlp.npz`
9. Prints a compression summary: original size, compressed size, and compression ratio

**Estimated time:** ~15–30 minutes on CPU, ~5–10 minutes on a GPU.

---

### Option 2: Run Stages Individually (Python API)

If you want to experiment with individual stages or use the project as a library, you can call each component separately from a Python script or Jupyter notebook:

```python
from config import config_device
from data.data_loader import CIFAR10_loader
from models.model_cifar import CompressionMLP
from utils.training import train_and_eval
from compression.pruning import prune_model
from compression.quantization import quantize_model
from compression.huffman import huffman_encode_model, print_compression_summary
from utils.loading import save_model_npz

# Setup — auto-detects cuda / mps / cpu
device = config_device()

# Load CIFAR-10 — downloads automatically on first run
train_loader, test_loader = CIFAR10_loader()

# Stage 0: Train the baseline MLP (no compression yet)
model = CompressionMLP().to(device)
train_and_eval(model, train_loader, test_loader, device, epochs=5)

# Stage 1: Prune 90% of weights by magnitude, then fine-tune to recover accuracy
prune_model(model, prune_ratio=0.90)
train_and_eval(model, train_loader, test_loader, device, epochs=2)

# Stage 2: Quantize remaining weights into 16 K-Means clusters
quantize_model(model, k=16)

# Stage 3: Huffman encode the cluster indices
huffman_encode_model(model)
print_compression_summary(model)

# Save to disk
save_model_npz(model, "compressed_models/compressed_mlp.npz")
```

---

## Expected Terminal Output

When running `python main.py`, here is roughly what you should see. If your output looks like this, everything is working correctly:

```
Using device: cuda        ← will say 'cpu' or 'mps' depending on your machine

--- Training Baseline MLP ---
Loading frozen feature extractor (AlexNet)...
CIFAR-10 loaded: 500 train, 100 test samples
Epoch [1/5] Average Loss: 2.2507
Epoch [2/5] Average Loss: 1.9781
Epoch [3/5] Average Loss: 1.6656
Epoch [4/5] Average Loss: 1.2838
Epoch [5/5] Average Loss: 1.0150
Evaluation Accuracy: 65.00%

--- Applying Pruning (90%) ---
Pruned: 37764223/41953280 (90.01% sparsity)
Fine-tuning pruned model...
Epoch [1/2] Average Loss: 1.1138
Epoch [2/2] Average Loss: 1.0129
Evaluation Accuracy: 64.00%

--- Applying Quantization (16 clusters) ---
Quantized classifier.1 to 16 clusters.
Quantized classifier.4 to 16 clusters.
Quantized classifier.6 to 16 clusters.
Quantization complete. 3 layers quantized.

--- Applying Huffman Encoding ---
Huffman encoded classifier.1: 3.88 bits/weight
Huffman encoded classifier.4: 3.97 bits/weight
Huffman encoded classifier.6: 4.00 bits/weight

Huffman encoding complete: 8.17x compression on indices 

============================================================
COMPRESSION SUMMARY
============================================================
Original model size: 228.00 MB
Compressed size: 17.96 MB
Compression ratio: 12.70x
============================================================

--- Serializing Model to NPZ ---
Model successfully serialized to compressed_models/compressed_mlp.npz
```

> Exact accuracy and compression numbers will vary slightly between runs due to random initialization.

---

## Loading the Model with Saved Weights

After `main.py` finishes, the compressed model is saved at `compressed_models/compressed_mlp.npz`. Use the following code to reload it later for inference — without re-running the full training pipeline:

```python
import torch
import torchvision.models as models
from models.model_cifar import CompressionMLP
from utils.loading import load_model_from_npz
from data.data_loader import CIFAR10_loader
from config import config_device

# Step 1: Setup device
device = config_device()

# Step 2: Create an empty model with the same architecture as during training
model = CompressionMLP().to(device)

# Step 3: Load compressed weights from the .npz file into the model
model = load_model_from_npz(model, "compressed_models/compressed_mlp.npz", device)

# Step 4: Always call .eval() before inference — disables dropout and batch norm training behaviour
model.eval()

# Step 5: Set up the same frozen AlexNet feature extractor used during training
feature_extractor = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).features
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# Step 6: Run inference on a batch of test images
_, test_loader = CIFAR10_loader()

with torch.no_grad():                                      # no_grad = don't compute gradients (faster, less memory)
    images, labels = next(iter(test_loader))               # grab one batch
    images = images.to(device)

    features = feature_extractor(images)                   # extract AlexNet features: shape [batch, 256, 6, 6]
    features = torch.flatten(features, 1)                  # flatten to [batch, 9216]
    outputs = model(features)                              # pass through compressed MLP: shape [batch, 10]
    _, predicted = torch.max(outputs, 1)                   # pick the class with highest score

print("Predicted:", predicted[:10])
print("Actual:   ", labels[:10])
```

**What `load_model_from_npz` does internally:**
- Reads the `.npz` file (a compressed NumPy archive)
- Restores pruning masks, cluster centers, and cluster indices back into each `ModifiedLinear` layer
- Reconstructs the quantized weight matrix from cluster data automatically
- Sets each layer's internal mode to `'quantized'` or `'pruned'` as appropriate, so the forward pass behaves correctly

**Important:** You must always pass images through the AlexNet feature extractor first. The model does not accept raw images directly — it only takes the 9216-dimensional feature vectors as input.

---

## Results

| Stage | Accuracy | Description |
|---|---|---|
| Baseline MLP | ~[fill after running]% | 5 epochs, no compression applied |
| After Pruning + Fine-tune | ~[fill after running]% | 90% weights zeroed, 2 epoch recovery |
| After Quantization (k=16) | ~[fill after running]% | 16 unique weight values per layer |
| Compressed size | ~[fill after running] MB | From `COMPRESSION SUMMARY` in terminal |
| Compression ratio | ~[fill after running]x | Original 228 MB ÷ Compressed size |

**How to fill in this table:** Run `python main.py` and read the terminal output. Accuracy after each stage is printed as `Evaluation Accuracy: XX.XX%`. The compressed size and ratio appear in the `COMPRESSION SUMMARY` block at the very end.

**What each stage contributes:**
- **Pruning** sets 90% of MLP weights to exactly zero, creating sparsity that can be stored compactly
- **Quantization** reduces all remaining weights to just 16 unique values (4 bits instead of 32 bits per weight)
- **Huffman coding** encodes each weight's cluster index using variable-length binary codes — more frequent indices get shorter codes, saving additional space

---

## Common Errors and Fixes

**`FileNotFoundError` when saving the model**
You forgot to create the output folder. Run `mkdir compressed_models` from the project root and try again.

**`ModuleNotFoundError: No module named 'torch'`**
Dependencies are not installed. Run `pip install torch torchvision scikit-learn numpy`. If you're using a virtual environment, make sure it's activated first (you should see `(venv)` in your terminal prompt).

**`ModuleNotFoundError: No module named 'data'` or `'models'` or `'compression'`**
You're running `main.py` from the wrong directory. Your terminal must be inside the project root — the folder that directly contains `main.py`. Run `cd <project-folder>` to get there first.

**Out of Memory / CUDA OOM**
Reduce the batch size. Open `config.py` and change `BATCH_SIZE = 128` to `BATCH_SIZE = 64` or `32`. AlexNet feature extraction is memory-intensive.

**Training is very slow on CPU**
This is expected — allow 15–30 minutes. If you have an Nvidia GPU, make sure you installed the CUDA version of PyTorch (see Setup Step 3). Run `python -c "import torch; print(torch.cuda.is_available())"` — if it prints `False` you're on CPU.

**Script freezes during quantization stage**
This is not a crash — K-Means (from scikit-learn) runs entirely on CPU and processes ~41 million weights. Expect a pause of 1–3 minutes per layer. Wait for it to finish.

**Poor accuracy after pruning (below 50%)**
This can happen with an unlucky random initialization. Try increasing fine-tuning epochs by editing the post-pruning `train_and_eval` call in `main.py` from `epochs=2` to `epochs=5`. Alternatively, reduce the pruning ratio from `0.90` to `0.85`.

---

## References

- Han et al., *Deep Compression* (2015) — [arXiv:1510.00149](https://arxiv.org/abs/1510.00149)
- Han et al., *Learning both Weights and Connections* (2015) — [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- Krizhevsky et al., *ImageNet Classification with Deep CNNs* (NeurIPS 2012)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

*EEA Project — April 2026*
