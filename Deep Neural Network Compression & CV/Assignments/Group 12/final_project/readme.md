```markdown
Fruits-360 Neural Network Compression Pipeline

A three-stage model compression pipeline — **L1 Pruning → K-Means Quantization → Huffman Coding** — applied to a custom CNN trained on the Fruits-360 dataset (131 classes, 100×100 RGB).  

Achieves **~9× storage compression** with **~3.9% Top-1 accuracy drop**. :contentReference[oaicite:0]{index=0}

---

```

```markdown
Repository Structure

project-root/
│
├── config.py                          # Device selection (CUDA / MPS / CPU)
├── main.py                            # End-to-end pipeline entry point
│
├── data/
│   ├── **init**.py
│   └── data_loader.py                 # FruitsDataset, get_dataloader, feature extraction
│
├── models/
│   ├── **init**.py
│   └── mnist.py                       # SmallCIFARNet architecture + mnist_model() factory
│
├── compression/
│   ├── **init**.py
│   ├── pruning.py                     # prune_model() + quantize_model() (re-exports)
│   ├── prune.py                       # L1 magnitude pruning implementation
│   ├── quantization.py                # K-Means quantization implementation
│   ├── linear.py                      # modified_linear (prunable + quantizable)
│   ├── conv2d.py                      # modified_conv2d (prunable + quantizable)
│   └── conv.py                        # modified_conv1d (available for extension)
│
├── utils/
│   ├── **init**.py
│   ├── training.py                    # train_one_epoch, evaluate, train_and_eval
│   ├── loading.py                     # save_model_npz, load_model_from_npz (Huffman)
│   └── test_eval.py                   # test_compressed_model, compare_models, benchmark
│
└── compressed_models/                 # Created automatically at runtime
├── model.pth                      # Stage 1 output: pruned float32 checkpoint
└── compressed.npz                 # Stage 3 output: Huffman-encoded archive

```

---

## ⚙️ How to Run

### Prerequisites

```bash
pip install torch torchvision numpy opencv-python scikit-learn
````

---

### 📦 Dataset Setup

Download **Fruits-360** from Kaggle and structure it like:

```
<your_data_path>/
    Training/
        Apple Braeburn/
        Banana/
        ...
    Test/
        Apple Braeburn/
        ...
```

---

### ▶️ Option A — Run Locally

Edit dataset path in `main.py`:

```python
# main.py
path = "/path/to/fruits-360"   # must contain Training/ and Test/
```

Run:

```bash
python main.py
```

This will:

* Train **SmallCIFARNet** for 10 epochs
* Apply **90% L1 pruning + fine-tuning (5 epochs)** → `model.pth`
* Apply **K-Means quantization (K=16) + 1 QAT epoch**
* Apply **Huffman encoding** → `compressed.npz`
* Reload and evaluate compressed model

---

### ☁️ Option B — Run on Kaggle

```python
BASE      = "/kaggle/input/datasets/moltean/fruits/fruits-360_100x100/fruits-360"
TRAIN_DIR = os.path.join(BASE, "Training")
TEST_DIR  = os.path.join(BASE, "Test")
```

Notebook automatically copies code to `/kaggle/working/`.

---

## 🔧 Component Guide

| Task                         | Location                          |
| ---------------------------- | --------------------------------- |
| Change dataset path          | `main.py` → `path`                |
| Change epochs                | `train_and_eval(..., epochs=N)`   |
| Change pruning sparsity      | `prune_model(model, 0.90)`        |
| Change quantization clusters | `quantize_model(model, 16)`       |
| Modify CNN                   | `models/mnist.py`                 |
| Add new layer type           | `compression/conv2d.py` (pattern) |
| Change image/batch size      | `main.py`                         |
| Inspect Huffman              | `utils/loading.py`                |
| Compare models               | `utils/test_eval.py`              |
| Benchmark latency            | `utils/test_eval.py`              |

---

## 📦 Loading the Compressed Model

```python
import torch
from models.mnist import mnist_model
from utils.loading import load_model_from_npz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mnist_model(num_classes=131, input_size=100)

model = load_model_from_npz(
    model,
    npz_path="compressed_models/compressed.npz",
    device=device,
)

model.eval()
with torch.no_grad():
    output = model(your_input_tensor)
    predictions = output.argmax(dim=1)
```

---

### 🔁 Load Intermediate Pruned Model

```python
model = mnist_model(num_classes=131, input_size=100)
model.load_state_dict(torch.load("compressed_models/model.pth", map_location=device))
model.to(device).eval()
```

---

## 📊 Results Summary

| Metric            | Value                    |
| ----------------- | ------------------------ |
| Dataset           | Fruits-360 (131 classes) |
| Baseline Top-1    | ~94.2%                   |
| Compressed Top-1  | ~90.3%                   |
| Baseline Top-5    | ~99.1%                   |
| Compressed Top-5  | ~98.1%                   |
| Original Size     | ~18.90 MB                |
| Compressed Size   | ~2.10 MB                 |
| Compression Ratio | ~9.0×                    |
| Space Saved       | ~88.9%                   |
| Accuracy Drop     | ~3.9%                    |

---

## 🧠 Compression Strategy

* **90% L1 Pruning**
* **K-Means Quantization (K=16)**
* **Huffman Encoding (lossless)**

> Note: Accuracy loss comes from pruning + quantization only.

---
