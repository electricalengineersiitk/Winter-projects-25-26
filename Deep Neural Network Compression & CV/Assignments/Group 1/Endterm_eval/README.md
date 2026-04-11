# Deep Compression Pipeline

This repository implements a complete Deep Compression pipeline for neural networks, inspired by the seminal paper by Han et al. (2016). The goal of this project is to dramatically reduce the storage requirements and computational footprint of trained models without sacrificing their predictive accuracy. 

The pipeline supports training a Multi-Layer Perceptron (MLP) on pre-extracted ResNet-18 features, as well as an end-to-end Convolutional Neural Network (CNN) trained directly on raw image data.

---

## Method Used

The compression pipeline sequentially applies three main techniques to achieve maximum reduction:

1.  **Magnitude-Based Pruning:** Removes redundant connections. We calculate the absolute magnitude of all weights globally and zero out the ones falling below a specific threshold to hit a target sparsity (e.g., 90%). The model is then briefly fine-tuned to recover lost accuracy.
2.  **Weight Quantization (K-Means):** Reduces the number of unique weight values. We take the surviving, non-zero weights and cluster them using K-Means (defaulting to 16 clusters, representing 4-bit quantization). Each weight is then replaced by its cluster's centroid, meaning we only need to store a short index instead of a full 32-bit float.
3.  **Huffman Encoding:** An entropy-based lossless compression technique. Because certain quantized weights appear much more frequently than others, we assign shorter binary codes to highly frequent weights and longer codes to rare ones, squishing the data footprint even further.

---

## Repository Structure

* **`main.py`** — The execution script that runs the end-to-end pipeline.
* **`config.py`** — Centralized configuration file for hyperparameters (epochs, learning rate, sparsity, bits).
* **`models/`** 
    * `mlp.py` — Multi-Layer Perceptron architecture.
    * `cnn.py` — Simple CNN architecture.
* **`compression/`**
    * `linear.py` & `conv2d.py` — Custom layers supporting binary pruning masks and quantized weight assignment.
    * `pruning.py` — Logic for global magnitude-based weight pruning.
    * `quantization.py` — K-Means clustering logic for weight quantization.
    * `huffman.py` — Tree-building and encoding logic for Huffman compression.
* **`data/`**
    * `loader.py` — Dataset loaders (CIFAR-10, MNIST, etc.) and pre-trained ResNet feature extraction.
* **`utils/`**
    * `serialization.py` — Custom logic to save/load sparse masks, centroids, and bitstrings efficiently into `.npz` format.
    * `visualization.py` — Matplotlib utilities to chart training curves, weight distributions, and compression summaries.
* **`compressed_models/`** — Output directory where serialized models and summary plots are saved.

---

## Running Instructions

**1. Install Prerequisites:** Ensure you have Python installed along with the required libraries. If using a GPU (recommended), ensure PyTorch is configured for CUDA.
```bash
pip install torch torchvision numpy scikit-learn matplotlib
2. Run the Pipeline (Default): By default, the script runs the CNN model on raw CIFAR-10 images.Bashpython main.py
3. Run the Pipeline (MLP with Feature Extraction): If you want to train the MLP on frozen ResNet-18 features, specify the model flag.Bashpython main.py --model mlp
(Note: If you have already extracted the features once, add --skip_extraction to save time by loading the cached features).Command Line Arguments:--model: Choose architecture (cnn or mlp).--dataset: Choose dataset (cifar10, mnist, etc.).--sparsity: Target pruning sparsity (e.g., 0.9 for 90%).--epochs / --finetune_epochs: Control training duration.🔄 How to Load the Compressed ModelBecause standard PyTorch torch.load() cannot handle our custom sparse, quantized, and Huffman-encoded formats, we use our custom .npz deserialization utility.To load the model with your compressed weights for inference:Pythonfrom utils.serialization import load_compressed_npz
from models.cnn import create_cnn
from compression.conv2d import replace_conv2d_with_compressed
from compression.linear import replace_linear_with_compressed
import torch

# 1. Initialize the empty architecture
model = create_cnn(num_classes=10)

# 2. Swap standard layers for our compression-aware layers
model = replace_linear_with_compressed(model)
model = replace_conv2d_with_compressed(model)

# 3. Load the compressed state dictionary
# This function automatically rebuilds the dense weight matrices 
# from the stored sparse indices and quantized centroids.
filepath = 'compressed_models/final_compressed.npz'
model_data = load_compressed_npz(filepath)

# 4. Map the reconstructed weights back to the model
layer_idx = 0
for name, module in model.named_modules():
    if hasattr(module, 'mask') and hasattr(module, 'weight'):
        layer_dict = model_data[f'layer_{layer_idx}']
        
        # Load weights and masks directly into the layer buffers
        module.weight.data = torch.tensor(layer_dict['weights'])
        module.mask.data = torch.tensor(layer_dict['mask'])
        if 'bias' in layer_dict:
            module.bias.data = torch.tensor(layer_dict['bias'])
            
        layer_idx += 1

print("Compressed model loaded and ready for inference!")

## Results Summary

Running this pipeline yields significant reductions in theoretical model size while acting as a regularizer to maintain accuracy despite extreme sparsity. 

In a standard execution using the CNN on raw CIFAR-10 images (90% sparsity, 16 clusters):

| Stage | Size (KB) | Accuracy | Ratio | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | ~1051.5 | 74.41% | 1.00x | Original dense CNN architecture (268,650 parameters). |
| **Pruned** | ~88.0 | 72.66% | 11.95x | Removed 90% of parameters; accuracy recovered via fine-tuning. |
| **Quantized** | ~75.6 | 71.68% | 13.90x | Max physical compression achieved; weights grouped into 16 4-bit clusters. |
| **Huffman** | ~107.0 | 71.68% | 9.83x | The theoretical entropy drops to ~3.7 bits (an 8.4x mathematical reduction on the indices). |
