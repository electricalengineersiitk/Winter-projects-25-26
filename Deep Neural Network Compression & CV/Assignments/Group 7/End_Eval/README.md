# Deep Compression of a Multimodal Fruit Classification Network

Implementation of the Deep Compression pipeline (Han et al., ICLR 2016) applied
to a custom multimodal neural network for 257-class fruit classification on the
Fruits-360 dataset. The model fuses LBP texture maps, Canny edge maps, and 12
handcrafted color/shape features — compressed via pruning, quantization, and
Huffman coding implemented from scratch.

---

## Project Structure

```
Final_Project/
│
├── main.py                        # Entry point — runs full pipeline
├── eval.py                        # Standalone evaluator for downloaded models
├── final_result_log.txt           # Full console output from the last pipeline run
├── config.py                      # Device selection (CUDA / MPS / CPU)
│
├── data/
│   ├── __init__.py
│   └── data_loader.py             # Dataset class, LBP, Canny, feature extraction
│
├── models/
│   ├── __init__.py
│   └── model_cifar.py             # SmallCIFARNet architecture
│
├── compression/
│   ├── __init__.py
│   ├── conv2d.py                  # modified_conv2d (prune + quantize aware)
│   ├── linear.py                  # modified_linear (prune + quantize aware)
│   ├── prune.py                   # prune_model(), count_sparsity()
│   ├── quantization.py            # quantize_model()
│   └── huffman.py                 # Huffman coding from scratch
│
├── utils/
│   ├── __init__.py
│   ├── training.py                # train_model(), evaluate(), train_and_eval()
│   ├── test_eval.py               # test_eval(), print_compression_table()
│   └── loading.py                 # save_model_npz(), load_model_from_npz()
│
└── compressed_models/             # Auto-created, stores all .npz checkpoints
    ├── baseline.npz
    ├── pruned.npz
    ├── quantized.npz
    ├── huffman.npz                # Huffman bitstreams + stats (0.33 MB)
    └── huffman_weights.npz        # Float16 weights for loading (~1.4 MB)
```

---

## How to Run

### 1. Install dependencies

```bash
pip install torch torchvision opencv-python numpy scikit-learn \
            kagglehub tqdm psutil
```

### 2. Run the full pipeline

```bash
python main.py
```

This will automatically:
- Download the Fruits-360 dataset via `kagglehub`
- Train the baseline model (10 epochs)
- Apply pruning + retrain (5 epochs)
- Apply quantization + retrain (5 epochs)
- Apply Huffman coding
- Save all models to `compressed_models/`
- Print the full compression results table

### 3. Evaluate a specific model (without retraining)

**[Download Project Zip](https://drive.google.com/file/d/1umRUdfIn9GKW41s-uExQcqxbJhTkP4S0/view?usp=sharing)**

The project zip includes `eval.py` for evaluating any checkpoint directly.
Open [`eval.py`](https://drive.google.com/file/d/1Nkr9rr2IToEppRi1CbDzugGTCie3D2uT/view?usp=sharing) and change the default path on **line 118** to whichever model
you want to test:

```python
# eval.py, line 118 — change filename to any of:
#   baseline.npz
#   pruned.npz
#   quantized.npz
#   huffman.npz          (loads weights from huffman_weights.npz automatically)
default = "compressed_models/quantized.npz",
```

Then run:

```bash
python eval.py
```

Or pass the model path directly as a flag:

```bash
python eval.py --model compressed_models/huffman.npz
```

> **Note on `huffman.npz`:** This file stores only the Huffman bitstreams and
> stats (0.33 MB). When you load it, `eval.py` automatically resolves weights
> from `huffman_weights.npz` in the same folder — both files must be present.

`eval.py` is already included in the project zip. It will:
- Download the Fruits-360 test set automatically via `kagglehub`
- Load the model from the `.npz` file
- Run evaluation on the full test set
- Print an accuracy + compression stats table

---

## Pre-trained Models

All models are included in the project zip — no separate downloads needed.
Just unzip and run `eval.py`.

**[Download Project Zip](https://drive.google.com/file/d/1umRUdfIn9GKW41s-uExQcqxbJhTkP4S0/view?usp=sharing)**

The `compressed_models/` folder inside the zip contains:

| File | Description | Size |
|---|---|---|
| `baseline.npz` | Fully trained, no compression | 47.30 MB |
| `pruned.npz` | 99.7% weights pruned + retrained | 4.83 MB |
| `quantized.npz` | k=16 quantization + retrained | 4.83 MB |
| `huffman.npz` | Huffman bitstreams + stats | 0.33 MB |
| `huffman_weights.npz` | Float16 weights (required alongside huffman.npz) | ~1.4 MB |

---

## Results

Full console output from the last pipeline run is saved in `final_result_log.txt`
at the project root.

```
------------------------------------------------------------
  DEEP COMPRESSION RESULTS
------------------------------------------------------------

  1. ACCURACY
  Baseline                              95.65%
  After Pruning  (before retrain)       40.79%   (loss 54.86%)
  After Pruning  (recovered)            95.24%   (Δ -0.41%)
  After Quantization (before retrain)   92.44%   (loss 2.80%)
  After Quantization (recovered)        95.29%   (Δ -0.37%)
  Total Accuracy Drop                   0.37%    within 1.5%

  2. STORAGE MEMORY
  Baseline (.npz)                       47.30 MB
  After Pruning (.npz)                   4.83 MB   (89.8% smaller   9.8x)
  After Quantization (.npz)              4.83 MB   (89.8% smaller   9.8x)
  Huffman Compressed (.npz)              0.33 MB   (99.3% smaller 141.5x)
  Final Compression Ratio                141.5x     ≥9x

  3. RUNTIME MEMORY
  GPU VRAM Used                         546.1 MB
  RAM Used by Process                   624.9 MB
  Total Parameters                    19,784,385
  Weights Pruned         19,715,118 / 19,781,920   (99.7% sparse)

  4. COMPRESSION DETAILS
  Quantization                          k=16 centroids   (~4-bit)
  Bits/weight after pruning+quant       0.0135
  Bits/weight after Huffman             1.0291           (target 3.57)
  Huffman coding gain                   15.55x over quantized

------------------------------------------------------------
```

| Metric | Target | Achieved |
|---|---|---|
| Compression ratio | ≥ 9× | **141.5×** |
| Accuracy drop | ≤ 1.5% | **0.37%** |
| Bits per weight | ≈ 3.57 | **1.0291** |
| Baseline accuracy | — | **95.65%** |

---

## Reference

> Song Han, Huizi Mao, William J. Dally.
> *Deep Compression: Compressing Deep Neural Networks with Pruning,
> Trained Quantization and Huffman Coding.*
> ICLR 2016. [arXiv:1510.00149](https://arxiv.org/abs/1510.00149)