# 🧠 EEG Brain Speller

> A complete Brain-Computer Interface (BCI) system that spells characters using P300 EEG brain signals — no physical movement required.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MNE](https://img.shields.io/badge/MNE-1.6%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📖 Table of Contents

1. [What is a P300 Brain Speller?](#what-is-a-p300-brain-speller)
2. [Project Structure](#project-structure)
3. [Libraries and Tools Used](#libraries-and-tools-used)
4. [Dataset](#dataset)
5. [Pipeline Overview](#pipeline-overview)
6. [Installation and Setup](#installation-and-setup)
7. [Running the Code](#running-the-code)
8. [Running on Google Colab](#running-on-google-colab)
9. [Results](#results)
10. [File Descriptions](#file-descriptions)
11. [References](#references)

---

## What is a P300 Brain Speller?

A P300 Brain Speller is a Brain-Computer Interface (BCI) that lets a person type characters using only their brain activity — no hands, no voice, no physical movement needed.

**How it works:**

1. A 6×6 matrix of 36 characters is shown on screen
2. Rows and columns flash in a random sequence
3. When the row/column containing the user's **target character** flashes, the brain produces a special brainwave called the **P300** — a positive voltage spike approximately 300 ms after the flash
4. EEG electrodes on the scalp record this signal
5. A machine learning classifier detects the P300 and figures out which character the user was focusing on

This technology is especially important for people with severe motor disabilities such as ALS, allowing them to communicate purely through brain signals.

---

## Project Structure

```
EEG/
├── src/
│   ├── __init__.py          # Package initialiser
│   ├── preprocess.py        # EEG filtering, ICA, epoching
│   ├── features.py          # Feature extraction (Xdawn, CSP, FFT, CCA, Riemannian)
│   ├── models.py            # LDA, SVM, EEGNet classifiers
│   ├── evaluate.py          # Cross-validation, ITR, plots, CSV export
│   └── speller_ui.py        # PsychoPy live stimulus interface
├── notebooks/
│   └── exploration.ipynb    # Interactive step-by-step notebook
├── data/                    # Raw and preprocessed EEG files (auto-downloaded)
├── results/                 # Saved plots, confusion matrices, metrics CSV
├── train_and_evaluate.py    # Main end-to-end training script
├── requirements.txt         # All Python dependencies
└── README.md                # This file
```

---

## Libraries and Tools Used

### Core EEG Processing

| Library | Version | What it does in this project |
|---|---|---|
| **MNE-Python** | >=1.6 | Loads EEG files, applies bandpass/notch filters, creates epochs, runs ICA for artefact removal, implements Xdawn spatial filtering |
| **MOABB** | >=0.5 | Downloads the BNCI2014-009 P300 dataset automatically, provides a clean API for BCI dataset benchmarking |
| **autoreject** | latest | Automatically detects and rejects bad EEG epochs using cross-validation |

### Signal Processing and Mathematics

| Library | Version | What it does in this project |
|---|---|---|
| **NumPy** | >=1.24 | Array operations, FFT computation, matrix math throughout the pipeline |
| **SciPy** | >=1.11 | Additional digital filter design, statistical functions |
| **pyriemann** | latest | Riemannian geometry-based covariance matrix features (advanced feature extraction option) |

### Machine Learning

| Library | Version | What it does in this project |
|---|---|---|
| **scikit-learn** | >=1.3 | LDA classifier, SVM classifier, stratified k-fold cross-validation, StandardScaler, train/test split, all evaluation metrics |
| **PyTorch** | >=2.0 | EEGNet deep learning model — depthwise separable CNN trained end-to-end on raw EEG epochs |
| **braindecode** | latest | Additional deep learning utilities for EEG, wraps PyTorch with EEG-specific tools |

### Visualisation and Data

| Library | Version | What it does in this project |
|---|---|---|
| **Matplotlib** | >=3.7 | ERP waveform plots, accuracy/ITR bar charts, all figure generation |
| **Seaborn** | >=0.12 | Confusion matrix heatmaps, styled statistical plots |
| **Pandas** | >=2.0 | Results logging, metrics CSV export, data management |

### Stimulus Delivery (Optional)

| Library | Version | What it does in this project |
|---|---|---|
| **PsychoPy** | >=2023.1 | Renders the live 6×6 character matrix with precise flash timing for real EEG experiments |

### Hardware Acceleration

| Tool | What it does |
|---|---|
| **CUDA (via PyTorch)** | GPU acceleration for EEGNet training on NVIDIA GPUs |
| **Apple MPS (via PyTorch)** | GPU acceleration for EEGNet on Apple M1/M2/M3 chips |
| **Google Colab T4 GPU** | Free cloud GPU used in this project for EEGNet training |

---

## Dataset

**Primary Dataset: BNCI2014-009 (EPFL P300 Speller)**

- **Paradigm:** P300 Speller
- **Subjects:** 10 healthy participants
- **Channels:** 16 EEG electrodes
- **Sampling rate:** 256 Hz
- **Trials per subject:** Multiple sessions with 36-character spelling tasks
- **Access:** Automatically downloaded via MOABB — no manual download needed

**Other supported datasets:**

| Dataset | Paradigm | Subjects | How to load |
|---|---|---|---|
| BCI Competition III Dataset II | P300 speller | 2 | Manual: https://www.bbci.de/competition/iii/ |
| Wang 2017 Benchmark | SSVEP | 35 | `moabb: Wang2016()` |
| BNCI2014-001 | Motor imagery | 9 | `moabb: BNCI2014_001()` |
| PhysioNet EEG Motor | Motor imagery | 109 | `mne.datasets.eegbci` |

---

## Pipeline Overview

```
Raw EEG Signal
      │
      ▼
┌─────────────────────────────────────┐
│  STAGE 1 — Preprocessing            │
│  • Bandpass filter: 0.1 – 30 Hz     │
│  • Notch filter: 50 Hz              │
│  • Average re-referencing           │
│  • Bad channel interpolation        │
│  • ICA artefact removal (blinks)    │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  STAGE 2 — Epoching                 │
│  • Window: -200 ms to +800 ms       │
│  • Baseline: -200 ms to 0 ms        │
│  • Locked to stimulus flash onset   │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  STAGE 3 — Feature Extraction       │
│  • Xdawn spatial filtering (P300)   │
│  • Downsample to 30 time points     │
│  • Feature vector: 360 values       │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  STAGE 4 — Classification           │
│  • LDA  — linear baseline           │
│  • SVM  — RBF kernel                │
│  • EEGNet — compact CNN (PyTorch)   │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  STAGE 5 — Evaluation               │
│  • 5-fold stratified CV             │
│  • Accuracy, Precision, Recall, F1  │
│  • ITR (bits/minute)                │
│  • Confusion matrix plots           │
└─────────────────────────────────────┘
```

**ITR Formula:**

```
ITR = ( log2(N) + P·log2(P) + (1−P)·log2((1−P)/(N−1)) ) × 60 / T

  N = 36  (number of symbols in the 6×6 matrix)
  P = classification accuracy as a fraction
  T = trial duration in seconds (2.0 s in this project)
```

---

## Installation and Setup

### Prerequisites

- Python 3.9 or higher (3.10 recommended)
- pip package manager
- Internet connection (for dataset download)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/EEG.git
cd EEG
```

### Step 2 — Create a virtual environment

```bash
# Create environment
python3 -m venv eeg_env

# Activate — Mac/Linux
source eeg_env/bin/activate

# Activate — Windows
eeg_env\Scripts\activate
```

You should see `(eeg_env)` in your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install --upgrade pip

# Core libraries
pip install mne numpy scipy scikit-learn matplotlib seaborn pandas moabb

# Deep learning
pip install torch torchvision torchaudio

# Advanced EEG tools
pip install braindecode pyriemann autoreject

# Stimulus delivery (optional — only needed for live EEG experiments)
pip install psychopy
```

### Step 4 — Verify installation

```bash
python3 -c "import mne, numpy, sklearn, moabb, torch; print('All good!')"
```

### Apple Silicon (M1/M2/M3) Note

PyTorch supports Apple's Metal GPU backend. Check if it's available:

```python
import torch
print(torch.backends.mps.is_available())  # True = GPU acceleration available
```

---

## Running the Code

### Option A — Interactive Notebook (Recommended for beginners)

```bash
pip install jupyter
jupyter notebook notebooks/exploration.ipynb
```

Run cells top to bottom. This walks you through every stage interactively and shows plots at each step.

### Option B — Full Pipeline Script

**Train LDA and SVM on subjects 1, 2, 3 with 5-fold CV:**

```bash
python train_and_evaluate.py --model lda --subject 1 2 3 --folds 5
python train_and_evaluate.py --model svm --subject 1 2 3 --folds 5
```

**Train EEGNet (requires GPU for reasonable speed):**

```bash
python train_and_evaluate.py --model eegnet --subject 1 2 3 --folds 5
```

**Run all models at once:**

```bash
python train_and_evaluate.py --model all --subject 1 2 3 --folds 5
```

**Command-line arguments:**

| Argument | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | `all` | Model to train: `lda`, `svm`, `eegnet`, `lda_flat`, or `all` |
| `--subject` | `-s` | first 3 | MOABB subject IDs (e.g. `--subject 1 2 3`) |
| `--folds` | `-k` | `5` | Number of cross-validation folds |

**Output files (saved to `results/`):**

| File | Description |
|---|---|
| `metrics.csv` | Accuracy, precision, recall, F1, ITR for all models |
| `accuracy_vs_itr.png` | Bar chart comparing models |
| `confusion_matrix_lda.png` | LDA confusion matrix heatmap |
| `confusion_matrix_svm.png` | SVM confusion matrix heatmap |
| `confusion_matrix_eegnet.png` | EEGNet confusion matrix heatmap |

### Option C — Live Stimulus Interface (PsychoPy)

Delivers the P300 flash stimulus to a real participant while recording EEG:

```bash
python src/speller_ui.py --target A --reps 10
```

| Argument | Description |
|---|---|
| `--target` | The character the user should focus on (default: A) |
| `--reps` | Number of row/column flash repetitions per trial (default: 10) |

---

## Running on Google Colab

### Step 1 — Upload to Google Drive

Upload your entire `EEG/` folder to Google Drive maintaining the folder structure.

### Step 2 — Open the notebook

Right-click `exploration.ipynb` in Drive → **Open with → Google Colaboratory**

### Step 3 — Enable GPU

**Runtime → Change runtime type → GPU (T4) → Save**

### Step 4 — Run setup cells

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/EEG')
```

```python
# Cell 2: Install packages (run every new session)
!pip install mne numpy scipy scikit-learn matplotlib seaborn pandas moabb
!pip install torch torchvision torchaudio
!pip install braindecode pyriemann autoreject
```

### Step 5 — Run the full pipeline

```python
from train_and_evaluate import run_experiment

run_experiment(model_names=["lda", "svm", "eegnet"], subject_ids=[1, 2, 3], k_folds=5)
```

### Step 6 — View results

```python
from IPython.display import Image
Image('/content/drive/MyDrive/EEG/results/accuracy_vs_itr.png')
```

> **Note:** PsychoPy is not supported in Colab (no display). Skip it — it is only needed for live EEG recording, not for the ML pipeline.

> **Note:** Re-run the pip install cell every time you start a new Colab session, as Colab resets the environment.

---

## Results

Results obtained on BNCI2014-009, subjects 1–3, 5-fold stratified cross-validation:

| Model | Accuracy | Precision | Recall | F1 Score | ITR (bits/min) |
|---|---|---|---|---|---|
| LDA | 84.86% | 0.7305 | 0.6341 | 0.6604 | 113.40 |
| SVM | 82.16% | 0.6762 | 0.6687 | 0.6723 | 107.35 |
| **EEGNet** | **89.59%** | **0.8159** | **0.8011** | **0.8082** | **124.61** |

**Key findings:**
- EEGNet outperforms classical methods by ~5% accuracy and ~11 bits/min ITR
- LDA outperforms SVM — consistent with published P300 literature
- All models exceed 80% accuracy and 100 bits/min ITR, well above typical BCI benchmarks

---

## File Descriptions

### `src/preprocess.py`
Full MNE preprocessing pipeline. Key functions:
- `preprocess_raw()` — convenience wrapper for the full pipeline
- `apply_bandpass()` — 0.1–30 Hz FIR filter
- `apply_notch()` — 50 Hz power-line noise removal
- `run_ica()` — automatic eye-blink artefact rejection
- `make_epochs()` — stimulus-locked epoch extraction with baseline correction

### `src/features.py`
Six sklearn-compatible feature extractors:
- `XdawnFeatures` — Xdawn spatial filtering for P300 (primary method)
- `EpochDownsampleFeatures` — simple waveform flattening baseline
- `SSVEPFFTFeatures` — FFT amplitude at stimulus frequencies for SSVEP
- `CCAFeatures` — Canonical Correlation Analysis for SSVEP
- `CSPFeatures` — Common Spatial Patterns for motor imagery
- `RiemannianFeatures` — Riemannian tangent-space features (requires pyriemann)

### `src/models.py`
Three classifiers with sklearn-compatible API:
- `build_lda()` — LDA wrapped in StandardScaler pipeline
- `build_svm()` — SVM (RBF kernel, class-balanced) pipeline
- `EEGNetClassifier` — full PyTorch EEGNet with fit/predict/predict_proba
- `RepetitionEnsemble` — averages soft scores across flash repetitions

### `src/evaluate.py`
Evaluation utilities:
- `compute_itr()` — ITR formula implementation
- `cross_validate_model()` — stratified k-fold CV
- `evaluate_on_test()` — full metrics on held-out test set
- `plot_confusion_matrix()` — styled heatmap
- `plot_accuracy_vs_itr()` — model comparison bar chart
- `save_metrics_csv()` — export results to CSV

### `src/speller_ui.py`
PsychoPy-based live stimulus interface:
- 6×6 character matrix with randomised row/column flashing
- Configurable flash duration (100 ms), ISI (75 ms), and repetitions
- Saves timestamped event log CSV for EEG synchronisation

### `train_and_evaluate.py`
Main entry point — orchestrates the full pipeline:
- Downloads BNCI2014-009 via MOABB
- Builds feature extraction + classifier pipelines
- Runs cross-validation and held-out test evaluation
- Saves all plots and metrics

### `notebooks/exploration.ipynb`
Interactive walkthrough:
1. Download and inspect dataset
2. Visualise raw EEG signals
3. Plot grand-average ERP (see the P300 bump at 300 ms)
4. Extract and inspect features
5. Train LDA and SVM, compare results
6. Plot confusion matrix

---

## References

- Farwell, L.A. & Donchin, E. (1988). Talking off the top of your head: toward a mental prosthesis utilizing event-related brain potentials. *Electroencephalography and Clinical Neurophysiology*, 70(6), 510–523.

- Lawhern, V.J. et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5).

- Lotte, F. et al. (2018). A review of classification algorithms for EEG-based brain-computer interfaces: A 10-year update. *Journal of Neural Engineering*, 15(3).

- Rivet, B. et al. (2009). xDAWN algorithm to enhance evoked potentials: Application to brain-computer interface. *IEEE Transactions on Biomedical Engineering*, 56(8), 2035–2043.

- Wang, Y. et al. (2017). A benchmark dataset for SSVEP-based brain-computer interfaces. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 25(10), 1746–1752.

---

## Useful Links

- MNE-Python documentation: https://mne.tools/stable/auto_tutorials/
- MOABB documentation: https://moabb.neurotechx.com/
- EEGNet GitHub: https://github.com/vlawhern/arl-eegmodels
- Braindecode library: https://braindecode.org/
- BCI Competition datasets: https://www.bbci.de/competition/

---

*Built for the EEG Brain Speller BCI Competition. Questions? Reach your mentor.*
