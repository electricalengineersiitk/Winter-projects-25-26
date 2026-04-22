# EEG Brain Speller
An EEG-based Brain-Computer Interface that lets a person type characters using only their brain signals. Built on the P300 speller paradigm using BCI Competition III Dataset II (Subjects A & B).

---

## How It Works
A 6×6 character grid flashes row by row and column by column. When the target character flashes, the brain produces a P300 response (~300 ms post-stimulus). The classifier detects this response across multiple repetitions, averaging scores per row and column to identify the intended character.

---

## Project Structure
eeg_speller/
├── data/
│   ├── Subject_A_Train.mat
│   ├── Subject_A_Test.mat
│   ├── Subject_B_Train.mat
│   └── Subject_B_Test.mat
├── src/
│   ├── dataloader.py     # Loads .mat files
│   ├── preprocess.py     # Bandpass/notch filtering + epoch extraction
│   ├── features.py       # Xdawn spatial filtering / EEGNet formatting
│   ├── models.py         # LDA, SVM, EEGNet + character decoding
│   ├── evaluate.py       # Accuracy, ITR, confusion matrix, k-fold CV
│   └── main.py           # Pipeline entry point
├── results/              # Saved confusion matrix plots
└── README.md
---

## Setup
*Python 3.10 recommended.*

```bash
python -m venv eeg_env
source eeg_env/bin/activate        # Linux/macOS
eeg_env\Scripts\activate           # Windows

pip install numpy scipy scikit-learn matplotlib seaborn
pip install torch torchvision torchaudio
pip install braindecode pyriemann
```

---

## Data
Download BCI Competition III Dataset II from https://www.bbci.de/competition/iii/ and place the four .mat files inside `data/`.

---

## Usage
Switch models by changing one line at the top of `src/main.py`:

```python
MODE = "lda"   # "lda" | "svm" | "eegnet"
```

Then run:

```bash
python src/main.py
```

The pipeline will:
1. Train on each subject's training data independently
2. Decode the test set at 3, 5, 7, 10, and 15 repetitions
3. Print character accuracy and ITR (bits/min) for each rep count
4. Save confusion matrices to `results/` (LDA and SVM only)

---

## Models
| Model | Description |
|-------|-------------|
| `lda` | Linear Discriminant Analysis — fast, best ITR after Xdawn filtering |
| `svm` | RBF kernel SVM — slower, marginally weaker than LDA on this dataset |
| `eegnet` | Compact CNN via braindecode — highest peak accuracy |

LDA and SVM use Xdawn spatial filtering (nfilter=6) before classification. EEGNet learns its own spatial filters internally and receives the raw downsampled epoch tensor.

---

## Evaluation Metrics
- **Character Accuracy** — fraction of correctly decoded characters on held-out test string
- **ITR (bits/min)** — Information Transfer Rate combining speed and accuracy
- **Precision / Recall / F1** — epoch-level P300 classification metrics (LDA/SVM only)
- **Stratified 5-Fold CV** — cross-validation on training set (LDA/SVM only)
