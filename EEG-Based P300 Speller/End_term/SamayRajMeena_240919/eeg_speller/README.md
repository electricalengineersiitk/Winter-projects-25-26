# 🧠 EEG Brain Speller

> A Brain-Computer Interface (BCI) system that enables a person to type characters using only their brainwave signals — no physical movement required.

**Author:** Samayraj Meena | **Roll No:** 240919
**Domain:** EEG Signal Processing · Brain-Computer Interface · Machine Learning
**Dataset:** BNCI2014-009 (EPFL P300 Speller) via MOABB — 10 Subjects
**Hardware:** Intel Core i5-1135G7 (CPU only — no GPU required)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [How It Works — P300 Paradigm](#how-it-works)
3. [Project Structure](#project-structure)
4. [Environment Setup](#environment-setup)
5. [Dataset](#dataset)
6. [Pipeline Stages](#pipeline-stages)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [References](#references)

---

## Project Overview

The EEG Brain Speller is a complete end-to-end BCI pipeline built on the **P300 Speller paradigm**. A 6×6 matrix of characters flashes row by row and column by column. When the target character's row or column flashes, the brain produces a distinctive EEG signal called the **P300 component** (~300 ms after the flash). The system detects this signal and predicts which character the user intended.

**Key features:**
- Automatic dataset download via MOABB (no manual setup needed)
- Full preprocessing: 0.1–20 Hz bandpass, 50 Hz notch filter, ICA artefact removal
- Epoch extraction with baseline correction (−200 to 0 ms)
- Temporal downsampling feature extraction (16 channels × 30 samples = 480 features)
- Classical ML classifiers: LDA and SVM with 5-fold stratified cross-validation
- Deep learning: EEGNet (Lawhern et al. 2018) — runs efficiently on CPU
- Evaluation: Accuracy, AUC-ROC, and ITR (bits/min)

---

## How It Works

### The P300 Paradigm

```
  Character Matrix (6×6)             P300 ERP Response

  ┌──────────────────────┐           Amplitude (µV)
  │  A   B   C   D   E  F│               ^
  │  G   H   I   J   K  L│               |      /\   ← P300 (~300ms)
  │  M   N   O   P   Q  R│   Target ─────|     /  \──────
  │  S   T   U   V   W  X│               |────/
  │  Y   Z   1   2   3  4│   Non-Target ──── flat line
  │  5   6   7   8   9  _│
  └──────────────────────┘           0  100 200 300 400 500 ms
           ↑
       Column flash
```

1. Each of the 12 rows/columns flashes randomly for ~125 ms
2. When the **target character's** row or column flashes → brain produces a **P300 response**
3. 10 repetitions are done per character to average out noise
4. The classifier scores all 12 rows/columns → row + column with highest P300 score → their intersection = predicted character

---

## Project Structure

```
eeg_speller/
│
├── data/                    # Raw EEG files (auto-downloaded by MOABB)
│
├── notebooks/               # Jupyter notebooks for exploration
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Filtering, ICA, epoching, artifact removal
│   ├── features.py          # Feature extraction (downsampling + flatten)
│   ├── models.py            # LDA, SVM, EEGNet classifiers
│   ├── evaluate.py          # Cross-validation, ITR calculation, plots
│   └── speller_ui.py        # Optional: Psychopy stimulus interface
│
├── results/                 # Saved plots, confusion matrices, ERP figures
├── models/                  # Saved trained model files (.pkl, .keras)
│
├── main.py                  # Full LDA + SVM pipeline runner
├── train_eegnet.py          # EEGNet training script
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Requirements

- Python 3.9 or higher (3.10 recommended)
- Works on standard consumer hardware — **no GPU required**

### Step 1 — Create virtual environment

```bash
python -m venv eeg_env

# Windows
eeg_env\Scripts\activate

# Linux / macOS
source eeg_env/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install mne numpy scipy scikit-learn matplotlib seaborn pandas moabb tensorflow joblib
```

### Dependencies

| Library | Version | Purpose |
|---|---|---|
| mne | >=1.6 | EEG preprocessing, ICA, epoching |
| numpy | >=1.24 | Array and signal math |
| scipy | >=1.11 | Butterworth filter, notch filter |
| scikit-learn | >=1.3 | LDA, SVM, cross-validation, metrics |
| moabb | >=0.5 | Dataset download and P300 paradigm |
| tensorflow | >=2.13 | EEGNet deep learning model |
| matplotlib | >=3.7 | ERP plots and training curves |
| seaborn | >=0.12 | Confusion matrix heatmaps |
| pandas | >=2.0 | Result logging |
| joblib | >=1.3 | Saving and loading sklearn models |

---

## Dataset

**BNCI2014-009 (EPFL P300 Speller)**

| Property | Details |
|---|---|
| Subjects | 10 healthy participants |
| Paradigm | P300 Speller (6×6 character matrix) |
| EEG Channels | 16 channels used in execution |
| Sampling Rate | 256 Hz |
| Epoch Window | −200 ms to +800 ms (257 samples) |
| Flash Duration | ~125 ms per flash |
| Flashes per Character | 12 (6 rows + 6 columns) |
| Repetitions | 10 repetitions per character |
| Access | `MOABB: BNCI2014_009()` — auto-downloaded |

```python
from moabb.datasets import BNCI2014_009
from moabb.paradigms import P300

dataset  = BNCI2014_009()
paradigm = P300(fmin=0.1, fmax=20.0, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.0))

# Returns epochs, labels, metadata for subject 1
X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1])
# X shape: (n_epochs, 16, 257)
y_binary = (y == 'Target').astype(int)
```
> Files are cached after the first download. No manual downloading required.

---

## Pipeline Stages
### Stage 1 — Preprocessing (`src/preprocess.py`)

| Step | Method | Parameters |
|---|---|---|
| Bandpass Filter | Butterworth IIR, zero-phase | 0.1–20 Hz, order 5 |
| Notch Filter | IIR Notch | 50 Hz, Q = 30 (Indian power line) |
| ICA Artefact Removal | FastICA via MNE | 10 components |
| Epoch Extraction | Flash onset detection (0→1) | −200 ms to +800 ms |
| Baseline Correction | Pre-stimulus mean subtraction | −200 to 0 ms |
| Output Shape | Per subject | `(n_epochs, 16, 257)` |

**Why ICA?** Eye blinks and muscle movements contaminate raw EEG. ICA decomposes the signal into independent components and removes non-cerebral artefacts before classification. NumPy arrays are wrapped back into MNE structures to run ICA, which stabilises classifier inputs against artefact contamination.

### Stage 2 — Feature Extraction (`src/features.py`)

**Method: Temporal Downsampling + Flattening**

The time axis is uniformly subsampled from 257 → 30 points, then all 16 channels are concatenated into a single feature vector:

```
Output shape: 16 channels × 30 samples = 480 features per epoch
```

```python
def downsample_flatten(X, target_samples=30):
    idx = np.linspace(0, X.shape[2]-1, target_samples, dtype=int)
    return X[:, :, idx].reshape(len(X), -1)   # shape: (n_epochs, 480)
```

All features are then standardised with `StandardScaler` (zero mean, unit variance) — critical for SVM's RBF kernel which uses Euclidean distances.

### Stage 3 — Classification (`src/models.py`)

**LDA — Linear Discriminant Analysis**
- Solver: SVD (numerically stable, avoids matrix inversion issues)
- Fast and interpretable — proved highly robust on downsampled temporal features
- Achieved the 2nd best test accuracy of 89.94%

**SVM — Support Vector Machine (RBF)**
- Non-linear decision boundaries via RBF kernel
- `class_weight='balanced'` — upweights target epochs to handle 1:5 imbalance
- `probability=True` — enables Platt scaling for character-level score averaging

**EEGNet — Deep Learning (Best Model)**
- Compact CNN operating directly on raw epochs — no manual feature extraction needed
- Input shape: `(batch, 16, 257, 1)`
- Runs efficiently on CPU — tested on Intel Core i5-1135G7

| Layer | Operation | Output Shape | Params |
|---|---|---|---|
| Input | EEG epochs | (16, 257, 1) | — |
| Conv2D (F1=8) | Temporal conv, kernel (1,64) | (16, 257, 8) | 512 |
| DepthwiseConv2D | Spatial filter (16,1), D=2 | (1, 257, 16) | 256 |
| AvgPool + Dropout | Pool (1,4), p=0.5 | (1, 64, 16) | — |
| SeparableConv2D | Pointwise conv, F2=16, (1,16) | (1, 64, 16) | 512 |
| AvgPool + Dropout | Pool (1,8), p=0.5 | (1, 8, 16) | — |
| Dense (Softmax) | 2-class output | (2,) | 258 |
| **Total** | | | **1,618** |

### Stage 4 — Evaluation (`src/evaluate.py`)

- **5-fold stratified cross-validation** on training set — preserves Target/Non-Target ratio per fold
- **80/20 stratified train/test split** — test set never seen during training
- **Accuracy** — proportion of correctly classified epochs
- **AUC-ROC** — primary metric for imbalanced data; AUC = 1.0 is perfect, 0.5 is random
- **ITR (bits/min)** — communication speed combining both accuracy and speed:

```
ITR = [ log2(N) + P·log2(P) + (1−P)·log2((1−P)/(N−1)) ] × (60/T)

  N = 36  (symbols in the 6×6 grid)
  P = classification accuracy (0–1)
  T = 15 s  (10 repetitions × 12 flashes × 0.125 s/flash)
```

---

## How to Run

### Full LDA + SVM pipeline — all 10 subjects

```bash
python main.py
```
### Specific subjects only

```bash
python main.py --subjects 1 2 3
```
### Skip re-downloading (after first run)

```bash
python main.py --skip-download
```
### Train EEGNet on a subject

```bash
python train_eegnet.py --subject 1
```
### Optional Psychopy speller UI demo (run locally only)

```bash
python src/speller_ui.py
```

> **Note:** The Psychopy interface is a demonstration only and has not been validated in a live real-time BCI session.

---

## Results

### 5-Fold Cross-Validation — Training Set

| Subject | LDA (Downsample) | SVM (Downsample) |
|:---:|:---:|:---:|
| S01 | 89.0% ± 2.0% | 84.2% ± 1.7% |
| S02 | 90.7% ± 2.2% | 88.5% ± 1.4% |
| S03 | 85.6% ± 2.4% | 86.1% ± 1.8% |
| S04 | 89.2% ± 1.2% | 88.3% ± 1.7% |
| S05 | 91.0% ± 1.2% | 89.4% ± 2.0% |
| S06 | 86.6% ± 2.0% | 82.6% ± 1.1% |
| S07 | 87.7% ± 1.7% | 83.9% ± 1.4% |
| S08 | 87.0% ± 1.3% | 85.2% ± 2.3% |
| S09 | 91.8% ± 2.2% | 90.0% ± 2.4% |
| S10 | 92.5% ± 1.5% | 92.7% ± 1.5% |
| **Average** | **89.1% ± 1.8%** | **87.1% ± 1.7%** |

### Test Set — Final Model Comparison

| Model | Input | Accuracy | AUC | ITR (bits/min) |
|---|---|:---:|:---:|:---:|
| LDA | Downsampled features | 89.94% | 0.9240 | 16.78 |
| SVM (RBF) | Downsampled features | 87.75% | 0.8876 | 16.04 |
| **EEGNet** | **Raw epochs** | **90.06%** | **—** | **16.82** |

### Key Findings

- **EEGNet** achieved the highest accuracy (90.06%) and ITR (16.82 bits/min), learning directly from raw epochs with no manual feature engineering.
- **LDA** (89.94%) outperformed SVM (87.75%) on test sets — its simple generative approach proved highly robust on downsampled temporal features despite the 1:5 class imbalance.
- **Inter-subject variability** is significant: S09 and S10 consistently exceeded 94% accuracy, while S03 and S08 were harder to decode (~83–85%).
- **ITR range** of 14.6–18.9 bits/min across subjects is consistent with published P300 BCI benchmarks.
- All results achieved on **CPU only** (Intel Core i5-1135G7) — demonstrating EEGNet's efficiency on standard consumer hardware.

### Output Files

```
results/
  ├── erp_subject_01.png          ← Average ERP: Target vs Non-Target
  ├── LDA_S01_confusion.png       ← LDA confusion matrix per subject
  ├── SVM_S01_confusion.png       ← SVM confusion matrix per subject
  └── eegnet_training_S01.png     ← EEGNet training accuracy & loss curves

models/
  ├── lda_subject_01.pkl          ← Saved LDA model
  ├── svm_subject_01.pkl          ← Saved SVM model
  └── eegnet_subject_01.keras     ← Saved EEGNet model
```

---

## References

1. Farwell, L.A. & Donchin, E. (1988). *Talking off the top of your head: toward a mental prosthesis utilizing event-related brain potentials.* Electroencephalography and Clinical Neurophysiology, 70(6), 510–523.

2. Lawhern, V.J. et al. (2018). *EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.* Journal of Neural Engineering, 15(5), 056013.

3. Lotte, F. et al. (2018). *A review of classification algorithms for EEG-based BCIs: a 10+ year update.* Journal of Neural Engineering, 15(3), 031005.

4. Jayaram, V. & Barachant, A. (2018). *MOABB: trustworthy algorithm benchmarking for BCIs.* Journal of Neural Engineering, 15(6), 066011.

5. Rivet, B. et al. (2009). *xDAWN algorithm to enhance evoked potentials: application to BCI.* IEEE Transactions on Biomedical Engineering, 56(8), 2035–2043.

6. Gramfort, A. et al. (2013). *MEG and EEG data analysis with MNE-Python.* Frontiers in Neuroscience, 7, 267.

---

*EEG Brain Speller · Samayraj Meena · Roll No: 240919 ·7 April 2026*
