# EEG Brain Speller

A Brain-Computer Interface (BCI) system that decodes EEG signals to identify intended characters from a 6×6 P300 speller matrix. Built on the BCI Competition III Dataset II (Subjects A & B).

---

## How It Works

A 6×6 character grid flashes row by row and column by column. When the target character flashes, the brain produces a P300 response (~300 ms after the flash). The classifier detects this response across multiple flash repetitions and scores each row and column to identify the intended character.

---

## Project Structure

```
eeg_speller/
├── data/
│   └── BCI_Comp_III_Wads_2004/
│       ├── Subject_A_Train.mat
│       ├── Subject_A_Test.mat
│       ├── Subject_B_Train.mat
│       └── Subject_B_Test.mat
├── src/
│   ├── load_data.py      # Load .mat files
│   ├── preprocess.py     # Epoch extraction
│   ├── features.py       # Xdawn spatial filtering / EEGNet formatting
│   ├── model.py          # LDA, SVM, EEGNet classifiers
│   ├── decode.py         # Row/column scoring and character decoding
│   ├── evaluate.py       # Accuracy, ITR, confusion matrix, k-fold CV
│   ├── speller_ui.py     # PsychoPy visual stimulus interface
│   └── main.py               # Full training + evaluation pipeline
├── readme.md
└── results
```

---

## Setup

**Python 3.10 recommended.**

```bash
python -m venv eeg_env
source eeg_env/bin/activate        # Linux/macOS
eeg_env\Scripts\activate           # Windows

pip install numpy scipy scikit-learn matplotlib seaborn
pip install torch torchvision torchaudio
pip install braindecode pyriemann
pip install psychopy
```

---

## Data

Download the BCI Competition III Dataset II from https://www.bbci.de/competition/iii/ and place the four `.mat` files in `data/BCI_Comp_III_Wads_2004/`.

---

## Ground Truth Strings

The true character sequences for Subjects A and B are obtained from the official BCI Competition III Dataset II website:

https://www.bbci.de/competition/iii/results/index.html

These strings are used as ground truth for evaluating character-level accuracy and computing Information Transfer Rate (ITR).

---

## Usage

Configure the run at the top of `src/main.py`:

```python
model_type = "eegnet"   # "lda" | "svm" | "eegnet"
UI_REPS    = 5          # number of flash repetitions used for decoding
SHOW_UI    = False      # set True to run the PsychoPy speller interface
```

Then run:

```bash
python -m src.main
```

The pipeline will:
1. Train on Subject A and B training data
2. Decode the test set at 3, 5, 7, 10, and 15 repetitions
3. Print character accuracy and ITR (bits/min) for each repetition count
4. Optionally launch the PsychoPy UI showing the decoded output

---

## Models

| Model | Description |
|-------|-------------|
| `lda` | Linear Discriminant Analysis — fast baseline |
| `svm` | Linear SVC with Platt calibration |
| `eegnet` | Compact CNN for EEG via braindecode |

LDA and SVM use Xdawn spatial filtering (6 filters) followed by flattening. EEGNet receives the raw downsampled epoch tensor directly.

---

## Evaluation Metrics

- **Character accuracy** — fraction of correctly decoded characters on the held-out test string
- **ITR (bits/min)** — Information Transfer Rate combining speed and accuracy
- **Precision / Recall / F1** — epoch-level P300 vs. non-P300 classification (LDA/SVM only)
- **Stratified 5-fold CV** — cross-validation on training data (LDA/SVM only)

---

## Results

### Character Accuracy (%)

| Subject | Model  | 3 reps | 5 reps | 7 reps | 10 reps | 15 reps |
|---------|--------|--------|--------|--------|---------|---------|
| A       | LDA    | 35%    | 54%    | 65%    | 76%     | 91%     |
| A       | SVM    | 39%    | 53%    | 68%    | 75%     | 92%     |
| A       | EEGNet | 40%    | 46%    | 61%    | 76%     | 85%     |
| B       | LDA    | 70%    | 82%    | 85%    | 88%     | 90%     |
| B       | SVM    | 72%    | 83%    | 85%    | 86%     | 91%     |
| B       | EEGNet | 64%    | 74%    | 86%    | 89%     | 95%     |

### ITR (bits/min)

| Subject | Model  | 3 reps | 5 reps | 7 reps | 10 reps | 15 reps |
|---------|--------|--------|--------|--------|---------|---------|
| A       | LDA    | 8.59   | 10.37  | 9.96   | 8.98    | 8.14    |
| A       | SVM    | 10.25  | 10.07  | 10.71  | 8.79    | 8.30    |
| A       | EEGNet | 10.68  | 8.03   | 9.00   | 8.98    | 7.22    |
| B       | LDA    | 26.19  | 20.38  | 15.47  | 11.50   | 7.98    |
| B       | SVM    | 27.41 | 20.80 | 15.47 | 11.05  | 8.14    |
| B       | EEGNet | 22.67  | 17.20  | 15.79  | 11.73   | 8.81    |

**Best accuracy:** 95% — EEGNet, Subject B, 15 reps  
**Peak ITR:** 27.41 bits/min — SVM, Subject B, 3 reps
