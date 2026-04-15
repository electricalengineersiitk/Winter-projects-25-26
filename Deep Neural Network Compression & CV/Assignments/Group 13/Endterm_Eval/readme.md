# Deep Compression: A 3-Stage Pipeline for Resource-Constrained Deployment

This repository implements the full **Deep Compression** pipeline—Pruning, Trained Quantization, and Huffman Coding—as described in the paper by Han et al. (2015). The goal is to shrink a VGG-style CNN (`SmallCIFARNet`) so it can fit entirely within on-chip SRAM, bypassing the energy-expensive "memory wall" of off-chip DRAM.

## 🚀 Key Features
* **Stage 1: Global L1 Pruning:** Achieves 90% sparsity by removing low-magnitude weights globally across the network.
* **Stage 2: Trained Quantization:** Implements K-Means clustering ($k=32$) with vectorized gradient aggregation for centroid fine-tuning.
* **Stage 3: Huffman Coding:** Lossless entropy encoding of quantized weights and relative sparse indices.
* **Hardware Optimized:** Native support for Apple Silicon (M1/M2/M3) via the PyTorch MPS backend.

---

## 🛠️ Installation & Setup

### 1. Download the Project
First, clone the repository to your local machine and navigate into the project directory:

```bash
git clone [https://github.com/SubZeroNewbie/Winter-projects-25-26.git](https://github.com/SubZeroNewbie/Winter-projects-25-26.git)
cd Winter-projects-25-26/Project
```
### 2. Environment Activation

```bash
# Create a virtual environment if you haven't already
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate 

# Install required dependencies
pip install torch torchvision numpy scikit-learn
```

### 3. Execution

To run the full end-to-end pipeline (Training -> Pruning -> Quantization -> Huffman):

⚠️ **Note**: This process includes fine-tuning epochs to recover accuracy loss after pruning and may take several minutes on an M1/M2 Mac.
```bash
python3 main.py
```
❗️**The below exection is much Faster** : If you have already run the pipeline and have a compressed.npz file in the compressed_models/ folder, you can **skip retraining and view the final evaluation metrics** instantly:

```bash
python3 print_results.py
```


## 📁 Project Structure
### compression/: Core optimization logic.

* prune.py: Magnitude-based global pruning.

* quantization.py: K-Means clustering and weight sharing.

* conv2d.py, linear.py: Custom wrappers handling mask and indices buffers.

* models/: model_cifar.py defines the VGG-style architecture.

### utils/:

* training.py: Standard loops + Vectorized Centroid Update logic for M1 performance.

* huffman.py: Stage 3 entropy encoding and relative stride calculation.

* loading.py: Custom .npz serialization to save actual disk space.

* performance.py: Benchmarks for latency (ms) and Peak RAM (MB).

* config.py: Centralized hyperparameters and device configuration.



## 📥 Loading the Compressed Model
Since the weights are stored in a custom sparse format, use the provided utility to load them:
```bash
from models.model_cifar import SmallCIFARNet
from utils.loading import load_model_from_npz
from config import DEVICE

# Initialize architecture
model = SmallCIFARNet().to(DEVICE)

# Load weights from the compressed .npz archive
load_model_from_npz(model, "compressed_models/compressed.npz", DEVICE)

model.eval()
```



## 📊 Final Results
Benchmarks performed on an 8GB M1 MacBook Air:

* Metric	Result
* Original Size (Dense)	**~2.54 MB**
* Compressed Size (Huffman)	**0.07 MB**
* Compression Ratio	**38.13x**
* Accuracy Change	**-0.98% (80.19% Final)**
* Avg Bits per Weight	**0.84 bits**
* Inference Latency	**0.47 ms / image**
* Peak RAM Usage	**~1033 MB**
